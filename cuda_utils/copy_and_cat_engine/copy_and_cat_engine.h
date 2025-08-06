#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

class CopyAndCatTensors {
    struct Deleter {
        void operator()(CopyAndCatTensors* ptr){ delete ptr;}
    };
    friend Deleter;

private:
    CopyAndCatTensors() = default;
    CopyAndCatTensors(const CopyAndCatTensors&) = delete;
    CopyAndCatTensors(const CopyAndCatTensors&&) = delete;
    CopyAndCatTensors& operator=(const CopyAndCatTensors&) = delete;
    CopyAndCatTensors& operator=(const CopyAndCatTensors&&) = delete;

    ~CopyAndCatTensors();

    static void destroyInstance(const uint32_t& key) {
        auto it = instances.find(key);
        if (it != instances.end()) {
            instances.erase(it);
        }
    }

    void _dispatchAsyncCopy(void* dst_ptr, const void* src_ptr, size_t size, cudaMemcpyKind dir, int group, bool lazy=false);

public:
    // Prevents instantiation outside the class and ensures unique instances.
    static std::shared_ptr<CopyAndCatTensors> getInstance(const uint32_t& key) {
        // Check if an instance with the given key exists.
        auto it = instances.find(key);
        if (it != instances.end()) {
            // If exists, return the existing instance.
            return it->second;
        } else {
            // If not, create a new instance, store it, and return it.
            auto instance = std::shared_ptr<CopyAndCatTensors>(new CopyAndCatTensors(), Deleter{});
            instances[key] = instance;
            return instance;
        }
    }
    
    void initCache();

private:
    static std::unordered_map<uint32_t, std::shared_ptr<CopyAndCatTensors>> instances;
    cudaStream_t d2hStream;
    cudaStream_t h2dStream;
    std::unordered_map<int, std::queue<std::function<void()>>> taskQueueMap;  // Copy task queue for lazy copy
    
public:
    void swapIn(const std::vector<torch::Tensor>& cpu_tensors, torch::Tensor& gpu_tensors, int group, bool lazy);
    // void swapOut(uint64_t cpuBlkId, uint64_t gpuBlkId);
    void waitGroupCompletion(int group);
};