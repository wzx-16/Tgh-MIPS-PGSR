#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "copy_and_cat_engine.h"

// Copy engine related apis
void init_cpu_cache(uint32_t pp_virtual_engine) {
    CopyAndCatTensors::getInstance(pp_virtual_engine)->initCache();
}

void swap_in(uint32_t pp_virtual_engine, 
             const std::vector<torch::Tensor>& cpu_tensors,
             torch::Tensor& gpu_tensors,
             int group,
             bool lazy = false) {
    CopyAndCatTensors::getInstance(pp_virtual_engine)->swapIn(cpu_tensors, gpu_tensors, group, lazy);
}

void wait_group_completion(uint32_t pp_virtual_engine, int group) {
    CopyAndCatTensors::getInstance(pp_virtual_engine)->waitGroupCompletion(group);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // pybind11::module offloading_ops =
    //     m.def_submodule("offloading_ops", "the apis of offloading");
    // // Copy engine related apis
    m.def("init_cpu_cache", &init_cpu_cache,
                        "init cpu cahe, each CacheEngine should come with a cpu cache",
                        pybind11::arg("pp_virtual_engine"));
    m.def("swapIn", &swap_in,
                        "swap_in",
                        pybind11::arg("pp_virtual_engine"),
                        pybind11::arg("cpu_tensors"),
                        pybind11::arg("gpu_tensors"),
                        pybind11::arg("group"),
                        pybind11::arg("lazy") = false);
    m.def("waitGroupCompletion", &wait_group_completion,
                        "wait a group of copy tasks to complete",
                        pybind11::arg("pp_virtual_engine"),
                        pybind11::arg("group"));
}