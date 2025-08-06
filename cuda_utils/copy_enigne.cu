#include "copy_engine.h"

#define DEFAULT_GROUP_ID -1

// Static member initialization
std::unordered_map<uint32_t, std::shared_ptr<mem_manager::AsyncCopyEngine>> mem_manager::AsyncCopyEngine::instances;

mem_manager::AsyncCopyEngine::~AsyncCopyEngine() {
    _running.store(false);
    swapThread.join();
    cudaStreamDestroy(h2dStream);
    cudaStreamDestroy(d2hStream);
}

void mem_manager::AsyncCopyEngine::waitGroupCompletion(int group) {
    // Flush previous copy
    cudaStreamSynchronize(d2hStream);
    cudaStreamSynchronize(h2dStream);

    // Issue next group lazy queue
    auto it = taskQueueMap.find(group + 1);
    if (it != taskQueueMap.end()) {
        std::queue<std::function<void()>>& queue = it->second;
        while (!queue.empty()) {
            auto task = std::move(queue.front());
            queue.pop();
            task();
        }
    }
}

void mem_manager::AsyncCopyEngine::initCache(std::vector<torch::Tensor> const& gpu_cache, uint64_t cpu_cache_size_gb, std::string const& attn_back, bool not_lazy)
{
    cudaStreamCreate(&d2hStream);
    cudaStreamCreate(&h2dStream);

    if (attn_back.find("FlashAttention-2") != std::string::npos) {
        cacheShape = KVCacheShape::KV_NBLKS;
    }
    else if (attn_back.find("Flashinfer") != std::string::npos) {
        cacheShape = KVCacheShape::NBLKS_KV;
    }
    else if (attn_back.find("Torch SDPA") != std::string::npos) {
        cacheShape = KVCacheShape::KV_NBLKS;
    }
    else if (attn_back.find("XFormers") != std::string::npos) {
        cacheShape = KVCacheShape::KV_NBLKS;
    }
    else {
        throw std::runtime_error("Unsupported attention backend: " + attn_back);
    }

    if (cacheShape == KVCacheShape::KV_NBLKS) {
        // shape of KV_NBLKS gpu_cache[0]: (2, num_blocks, ...)
        singleLayerBlockSizeInBytes = gpu_cache[0].element_size() * gpu_cache[0][0][0].numel();
        kvStride = gpu_cache[0].element_size() * gpu_cache[0][0].numel();
        blockStride = singleLayerBlockSizeInBytes;
        blockNumGPU = gpu_cache[0].sizes()[1];
        cacheLayerNum = gpu_cache.size();
    }
    else {
        // shape of NBLKS_KV gpu_cache[0]: (num_blocks, 2, ...)
        singleLayerBlockSizeInBytes = gpu_cache[0].element_size() * gpu_cache[0][0].numel() / 2;
        kvStride = 0;
        blockStride = 2 * singleLayerBlockSizeInBytes;
        blockNumGPU = gpu_cache[0].sizes()[0];
        cacheLayerNum = gpu_cache.size();
    }

    std::vector<void*> _gpuCachePtrs;
    for (auto const& tensor : gpu_cache) {
        _gpuCachePtrs.push_back(tensor.data_ptr());
    }
    _memPoolGPU = std::make_unique<AbtractGPUMemPool>(_gpuCachePtrs, blockNumGPU, blockStride);

    // Allocate CPU cache, layout: (num_blocks, 2, ...)
    blockNumCPU = cpu_cache_size_gb * 1024 * 1024 * 1024 / (2 * singleLayerBlockSizeInBytes * cacheLayerNum);
    _memPoolCPU = std::make_unique<HostMemPool>(blockNumCPU, 2 * singleLayerBlockSizeInBytes * cacheLayerNum, cudaHostAllocWriteCombined);
    
    // if lazy launch in layer 1 ~ n
    notLazy = not_lazy;

    // launch swap consumer
    _running.store(true);
    _mpSwapQueue.reset(shmmqueue::CMessageQueue::CreateInstance(
        SHAR_KEY_BASE + 10 * _rank, 1 << 16, shmmqueue::eQueueModel::ONE_READ_ONE_WRITE));
    _launchSwapConsumer();

#ifdef TACO_LLM_PRINT_LOG
    TACO_LLM_LOG(DEBUG) << "[init AsyncCopyEngine] kvStride: " << kvStride << " "
                        << "blockStride: " << blockStride << " "
                        << "blockNumGPU: " << blockNumGPU << " "
                        << "blockNumCPU: " << blockNumCPU << " "
                        << "singleLayerBlockSizeInBytes: " << singleLayerBlockSizeInBytes << " "
                        << "cacheLayerNum: " << cacheLayerNum << " "
                        << "notLazy: " << std::boolalpha << notLazy << std::endl; 
#endif
}

void mem_manager::AsyncCopyEngine::swapIn(uint64_t cpuBlkId, uint64_t gpuBlkId)
{
    // Copy from CPU to GPU
    std::vector<void*> _gpuPtrs;
    _memPoolGPU->getSpecifiedBlockPtr(gpuBlkId, _gpuPtrs);
    for (int i = 0; i < cacheLayerNum; i++) {
        bool lazy = (notLazy || i == 0) ? false : true;
        if (cacheShape == KVCacheShape::KV_NBLKS) {
            _dispatchAsyncCopy(_gpuPtrs[i],
                               (void*)((uint64_t)_memPoolCPU->getSpecifiedBlockPtr(cpuBlkId) + 2 * i * singleLayerBlockSizeInBytes),
                               singleLayerBlockSizeInBytes, cudaMemcpyHostToDevice , i, lazy);
            _dispatchAsyncCopy((void*)((uint64_t)_gpuPtrs[i] + kvStride),
                               (void*)((uint64_t)_memPoolCPU->getSpecifiedBlockPtr(cpuBlkId) + (2 * i + 1) * singleLayerBlockSizeInBytes),
                               singleLayerBlockSizeInBytes, cudaMemcpyHostToDevice, i, lazy);
        }
        else {
            _dispatchAsyncCopy(_gpuPtrs[i],
                               (void*)((uint64_t)_memPoolCPU->getSpecifiedBlockPtr(cpuBlkId) + 2 * i * singleLayerBlockSizeInBytes),
                               singleLayerBlockSizeInBytes * 2, cudaMemcpyHostToDevice, i, lazy);
        }
    }
}

void mem_manager::AsyncCopyEngine::swapOut(uint64_t cpuBlkId, uint64_t gpuBlkId)
{
    // Copy from GPU to CPU
    std::vector<void*> _gpuPtrs;
    _memPoolGPU->getSpecifiedBlockPtr(gpuBlkId, _gpuPtrs);
    for (int i = 0; i < cacheLayerNum; i++) {
        bool lazy = (notLazy || i == 0) ? false : true;
        if (cacheShape == KVCacheShape::KV_NBLKS) {
            _dispatchAsyncCopy((void*)((uint64_t)_memPoolCPU->getSpecifiedBlockPtr(cpuBlkId) + 2 * i * singleLayerBlockSizeInBytes),
                               _gpuPtrs[i], singleLayerBlockSizeInBytes, cudaMemcpyDeviceToHost, i, lazy);
            _dispatchAsyncCopy((void*)((uint64_t)_memPoolCPU->getSpecifiedBlockPtr(cpuBlkId) + (2 * i + 1) * singleLayerBlockSizeInBytes),
                               (void*)((uint64_t)_gpuPtrs[i] + kvStride),
                               singleLayerBlockSizeInBytes, cudaMemcpyDeviceToHost, i, lazy);
        }
        else {
            _dispatchAsyncCopy((void*)((uint64_t)_memPoolCPU->getSpecifiedBlockPtr(cpuBlkId) + 2 * i * singleLayerBlockSizeInBytes),
                               _gpuPtrs[i], singleLayerBlockSizeInBytes * 2, cudaMemcpyDeviceToHost, i, lazy);
        }
    }
}

void mem_manager::AsyncCopyEngine::_launchSwapConsumer()
{
    swapThread = std::thread([&]() {
        constexpr size_t _msgSize = sizeof(CopyTaskDescriptor);
        unsigned char _msg[_msgSize] = {0};
        while (_running.load()) {
            int len = _mpSwapQueue->GetMessage(_msg);
            if (len > 0) {
                CopyTaskDescriptor* _task = (CopyTaskDescriptor*)_msg;
                uint64_t _srcBlkId = _task->srcBlkId;
                uint64_t _tarBlkId = _task->tarBlkId;
                cudaMemcpyKind _swapType = _task->dir;
                // std::cout << "worker " << _rank << " swap " << _srcBlkId << " to " << _tarBlkId << std::endl;
                if (_swapType == cudaMemcpyHostToDevice) {
                    swapIn(_srcBlkId, _tarBlkId);
                }
                else if (_swapType == cudaMemcpyDeviceToHost) {
                    swapOut(_tarBlkId, _srcBlkId);
                }
                else {
                    TACO_LLM_LOG(ERROR) << "Unsupported swap type: " << _swapType << std::endl;
                }
            }
            else {
                if (len != (int)shmmqueue::eQueueErrorCode::QUEUE_NO_MESSAGE) {
                    printf("Read failed ret = %d\n", len);
                    _mpSwapQueue->PrintTrunk();
                    continue;
                }
            }
        }
    });
}

void mem_manager::AsyncCopyEngine::_dispatchAsyncCopy(void* dst_ptr, const void* src_ptr, size_t size, cudaMemcpyKind dir, int group, bool lazy)
{
    if (lazy) {
        taskQueueMap[group].push([this, dst_ptr, src_ptr, size, dir]() {
            cudaMemcpyAsync(dst_ptr, src_ptr, size, dir, (dir == cudaMemcpyDeviceToHost)? this->d2hStream : this->h2dStream);
        });
    } else {
        cudaMemcpyAsync(dst_ptr, src_ptr, size, dir, (dir == cudaMemcpyDeviceToHost)? d2hStream : h2dStream);
    }
}
