#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

/*
dx * x + dy * y = z(a, b) - z(0, 0);
*/
__global__ void plane_fitting_cuda_foward_kernel(
    float sigma,
    float min_disp,
    float max_disp,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> random,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output)
{
    const int N = input.size(0);
    const int H = input.size(1);
    const int W = input.size(2);
    const int L = input.size(3);
    const int I = random.size(2);

    const int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= N * L)
    {
        return;
    }
    const int n = Index / L;
    const int l = Index % L;

    int max_inlier = 0;
    float best_dx = 0.0f;
    float best_dy = 0.0f;
    float z00 = input[n][H / 2][W / 2][l];
    if (z00 < min_disp || z00 > max_disp)
    {
        output[n][0][l] = best_dx;
        output[n][1][l] = best_dy;
        return;
    }

    for (int i = 0; i < I; ++i)
    {
        int ids0 = random[n][l][i][0];
        ids0 = (ids0 >= H * W / 2) ? ids0 + 1 : ids0;
        int x0 = ids0 % W;
        int y0 = ids0 / W;
        float z0 = input[n][y0][x0][l];
        if (z0 < min_disp || z0 > max_disp)
        {
            continue;
        }

        int ids1 = random[n][l][i][1];
        ids1 = (ids1 >= H * W / 2) ? ids1 + 1 : ids1;
        int x1 = ids1 % W;
        int y1 = ids1 / W;
        float z1 = input[n][y1][x1][l];
        if (z1 < min_disp || z1 > max_disp)
        {
            continue;
        }

        x0 -= (W / 2);
        y0 -= (H / 2);

        x1 -= (W / 2);
        y1 -= (H / 2);

        float c0 = z0 - z00;
        float c1 = z1 - z00;

        float dx = (c0 * y1 - y0 * c1) / (x0 * y1 - y0 * x1);
        float dy = (x0 * c1 - c0 * x1) / (x0 * y1 - y0 * x1);

        int inlier = 0;
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                float zwh = input[n][h][w][l];
                if (zwh < min_disp || zwh > max_disp)
                {
                    continue;
                }

                float err = dx * (w - W / 2) + dy * (h - H / 2) - zwh + z00;
                if (err < 0)
                {
                    err = -err;
                }

                if (err < sigma)
                {
                    ++inlier;
                }
            }
        }

        if (inlier > max_inlier)
        {
            max_inlier = inlier;
            best_dx = dx;
            best_dy = dy;
        }
    }

    output[n][0][l] = best_dx;
    output[n][1][l] = best_dy;
    return;
}

torch::Tensor plane_fitting_cuda_foward(torch::Tensor input, int iter, float sigma, float min_disp, float max_disp)
{
    const auto N = input.size(0);
    const auto H = input.size(1);
    const auto W = input.size(2);
    const auto L = input.size(3);

    torch::Tensor random = torch::randint(0, H * W - 1, {N, L, iter, 2}, torch::dtype(torch::kInt32).device(input.device()));
    torch::Tensor output = torch::ones({N, 2, L}, torch::dtype(torch::kFloat32).device(input.device()));

    const at::cuda::OptionalCUDAGuard guard(device_of(input));
    plane_fitting_cuda_foward_kernel<<<(N * L + 1023) / 1024, 1024>>>(
        sigma,
        min_disp,
        max_disp,
        input.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        random.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    return output;
}