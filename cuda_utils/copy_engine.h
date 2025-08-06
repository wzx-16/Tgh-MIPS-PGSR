# pragma once

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "offload_mem_pool.h"
#include "shmqueue/shmmqueue.h"

#define DEFAULT_GROUP_ID -1
#define SHAR_KEY_BASE 1000000

namespace mem_manager {

enum class KVCacheShape { KV_NBLKS, NBLKS_KV, KV_CACHE_SHAPE_NUM };

struct CopyTaskDescriptor {
    uint64_t srcBlkId;
    uint64_t tarBlkId;
    cudaMemcpyKind dir;
};

class AsyncCopyEngine {
    struct Deleter {
        void operator()(AsyncCopyEngine* ptr){ delete ptr;}
    };
    friend Deleter;

private:
    AsyncCopyEngine() = default;
    AsyncCopyEngine(const AsyncCopyEngine&) = delete;
    AsyncCopyEngine(const AsyncCopyEngine&&) = delete;
    AsyncCopyEngine& operator=(const AsyncCopyEngine&) = delete;
    AsyncCopyEngine& operator=(const AsyncCopyEngine&&) = delete;

    ~AsyncCopyEngine();

    static void destroyInstance(const uint32_t& key) {
        auto it = instances.find(key);
        if (it != instances.end()) {
            instances.erase(it);
        }
    }

public:
    // Prevents instantiation outside the class and ensures unique instances.
    static std::shared_ptr<AsyncCopyEngine> getInstance(const uint32_t& key) {
        // Check if an instance with the given key exists.
        auto it = instances.find(key);
        if (it != instances.end()) {
            // If exists, return the existing instance.
            return it->second;
        } else {
            // If not, create a new instance, store it, and return it.
            auto instance = std::shared_ptr<AsyncCopyEngine>(new AsyncCopyEngine(), Deleter{});
            instances[key] = instance;
            return instance;
        }
    }

    // attn_back: "flash_attn", "flash_infer", "torch_sdpa", "xformers"
    void initCache(std::vector<torch::Tensor> const& gpu_cache, uint64_t cpu_cache_size_gb, std::string const& attn_back, bool not_lazy);
    void setRank(uint32_t rank) { _rank = rank; }

    // offload: true for offloading, false for onloading
    void waitGroupCompletion(int group = DEFAULT_GROUP_ID);

    inline size_t getBlockNumGPU() const { return blockNumGPU; }
    inline size_t getBlockNumCPU() const { return blockNumCPU; }
    inline size_t getCacheLayerNum() const { return cacheLayerNum; }
    inline size_t getSingleLayerBlockSizeInBytes() const { return singleLayerBlockSizeInBytes; }

private:
    void swapIn(uint64_t cpuBlkId, uint64_t gpuBlkId);
    void swapOut(uint64_t cpuBlkId, uint64_t gpuBlkId);

    void _launchSwapConsumer();
    void _dispatchAsyncCopy(void* dst_ptr, const void* src_ptr, size_t size, cudaMemcpyKind dir, int group = DEFAULT_GROUP_ID, bool lazy=false);

private:
    std::atomic_bool _running = false;
    uint32_t _rank;

    size_t blockNumGPU;
    size_t blockNumCPU;
    size_t cacheLayerNum;
    size_t singleLayerBlockSizeInBytes;  // size of K/V
    size_t kvStride;
    size_t blockStride;  // stride of gpu block

    KVCacheShape cacheShape = KVCacheShape::KV_CACHE_SHAPE_NUM;

    cudaStream_t d2hStream;
    cudaStream_t h2dStream;
    std::unordered_map<int, std::queue<std::function<void()>>> taskQueueMap;  // Copy task queue for lazy copy
    bool notLazy = false;

    // Shared memory task queue for launching swap tasks, (src, dst, direction)
    std::shared_ptr<shmmqueue::CMessageQueue> _mpSwapQueue = nullptr;
    std::thread swapThread;

    std::unique_ptr<HostMemPool> _memPoolCPU = nullptr;
    std::unique_ptr<AbtractGPUMemPool> _memPoolGPU = nullptr;

    // Multiple instances for pipeline-parallism support.
    static std::unordered_map<uint32_t, std::shared_ptr<AsyncCopyEngine>> instances;
};
} // namespace mem_manager