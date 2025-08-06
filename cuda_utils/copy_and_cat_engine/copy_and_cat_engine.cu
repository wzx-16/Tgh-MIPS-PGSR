#include "copy_and_cat_engine.h"

// Static member initialization
std::unordered_map<uint32_t, std::shared_ptr<CopyAndCatTensors>> CopyAndCatTensors::instances;

CopyAndCatTensors::~CopyAndCatTensors() {
    cudaStreamDestroy(h2dStream);
    cudaStreamDestroy(d2hStream);
}

void CopyAndCatTensors::initCache()
{
    cudaStreamCreate(&d2hStream);
    cudaStreamCreate(&h2dStream);
}

void CopyAndCatTensors::waitGroupCompletion(int group) {
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

void CopyAndCatTensors::_dispatchAsyncCopy(void* dst_ptr, const void* src_ptr, size_t size, cudaMemcpyKind dir, int group, bool lazy)
{
    if (lazy) {
        taskQueueMap[group].push([this, dst_ptr, src_ptr, size, dir]() {
            cudaMemcpyAsync(dst_ptr, src_ptr, size, dir, (dir == cudaMemcpyDeviceToHost)? this->d2hStream : this->h2dStream);
        });
    } else {
        cudaMemcpyAsync(dst_ptr, src_ptr, size, dir, (dir == cudaMemcpyDeviceToHost)? d2hStream : h2dStream);
    }
}

void CopyAndCatTensors::swapIn(const std::vector<torch::Tensor>& cpu_tensors, torch::Tensor& gpu_tensors, int group, bool lazy)
{
    // Copy from CPU to GPU
    int offset = 0;
    for (const auto& tensor : cpu_tensors) {
        _dispatchAsyncCopy(gpu_tensors.data_ptr() + offset,
                            tensor.data_ptr(),
                            tensor.nbytes(), cudaMemcpyHostToDevice, group, lazy);
        offset += tensor.nbytes();
    }
}

// void CopyAndCatTensors::swapIn(const std::vector<torch::Tensor>& cpu_tensors, torch::Tensor& gpu_tensors)
// {
//     // Copy each tensor to the right position in the output
//     int offset = 0;
//     for (const auto& tensor : cpu_tensors) {
//         // Direct copy from CPU to the proper slice of GPU tensor
//         cudaMemcpyAsync(gpu_tensors.data_ptr() + offset, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyHostToDevice, h2dStream);
//         offset += tensor.nbytes();
//     }
// }

// void CopyAndCatTensors::swapOut(uint64_t cpuBlkId, uint64_t gpuBlkId)
// {

// }