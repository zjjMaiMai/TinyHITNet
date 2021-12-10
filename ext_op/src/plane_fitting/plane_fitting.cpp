#include <torch/extension.h>

torch::Tensor plane_fitting_cuda_foward(torch::Tensor input, int iter, float sigma, float min_disp, float max_disp);

torch::Tensor plane_fitting_foward(torch::Tensor input, int iter, float sigma, float min_disp, float max_disp)
{
    return plane_fitting_cuda_foward(input, iter, sigma, min_disp, max_disp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("plane_fitting_foward", &plane_fitting_foward, "plane_fitting_foward");
}