#include "layer/integral.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void IntegralRowKernel(const double *in, const std::uint32_t w, const std::uint32_t h, double *out)
    {
        const int tid = GET_THREAD_ID();
        if (tid < h)
        {
            double rs = 0;
            for (int c = 0; c < w; c++)
            {
                rs += in[tid * w + c];
                out[tid * w + c] = rs;
            }
        }
    }

    __global__ void IntegralColumnKernel(double *in, const std::uint32_t w, const std::uint32_t h)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < w)
        {
            double rs = in[tid];
            for (std::uint32_t r = 1; r < h; r++)
            {
                rs += in[r * w + tid];
                in[r * w + tid] = rs;
            }
        }
    }

    // __global__ void IntegralRowKernelS1(const std::uint32_t N, const double *in, double *out)
    // {
    //     const std::uint32_t tid = GET_THREAD_ID();
    //     if (tid < N)
    //     {
    //         std::uint32_t offset = tid * 4;

    //         out[offset] = in[offset];
    //         out[offset + 1] = in[offset] + in[offset + 1];
    //         out[offset + 2] = in[offset + 1] + in[offset + 2];
    //         out[offset + 3] = in[offset + 2] + in[offset + 3];
    //     }
    // }

    // __global__ void IntegralRowKernelS2(double *in, const std::uint32_t w, const std::uint32_t h)
    // {
    //     const std::uint32_t tid = GET_THREAD_ID();
    //     if (tid < h)
    //     {
    //         std::uint32_t offset = tid * w;
    //         for (std::uint32_t i = 7; i < w; i += 4)
    //         {
    //             in[offset + i] += in[offset + i - 4];
    //         }
    //     }
    // }

    // __global__ void IntegralRowKernelS3(const std::uint32_t N, double *in, const std::uint32_t w)
    // {
    //     const std::uint32_t tid = GET_THREAD_ID();
    //     if (tid < N && (tid * 4) % w > 0)
    //     {
    //         std::uint32_t offset = tid * 4;
    //         double val = in[offset - 1];

    //         in[offset] += val;
    //         in[offset + 1] += val;
    //         in[offset + 2] += val;
    //     }
    // }

    // __global__ void IntegralColumnKernelS1(const std::uint32_t N, double *in, const std::uint32_t w)
    // {
    //     const std::uint32_t tid = GET_THREAD_ID();
    //     if (tid < N)
    //     {
    //         std::uint32_t offset = tid / w * w * 4 + tid % w;

    //         in[offset + w] += in[offset];
    //         in[offset + 2 * w] += in[offset + 1 * w];
    //         in[offset + 3 * w] += in[offset + 2 * w];
    //     }
    // }

    // __global__ void IntegralColumnKernelS2(double *in, const std::uint32_t w, const std::uint32_t h)
    // {
    //     const std::uint32_t tid = GET_THREAD_ID();
    //     if (tid < w)
    //     {
    //         for (std::uint32_t i = 7; i < h; i += 4)
    //         {
    //             in[tid + i * w] += in[tid + (i - 4) * w];
    //         }
    //     }
    // }

    // __global__ void IntegralColumnKernelS3(const std::uint32_t N, double *in, const std::uint32_t w)
    // {
    //     const std::uint32_t tid = GET_THREAD_ID();
    //     if (tid < N && tid / w > 0)
    //     {
    //         const std::uint32_t offset = tid / w * w * 4 + tid % w;
    //         double val = in[offset - w];

    //         in[offset] += val;
    //         in[offset + 1] += val;
    //         in[offset + 2] += val;
    //     }
    // }

    void Integral::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
    {
        CHECK_EQ(bottoms.size(), 1);
        CHECK_EQ(tops.size(), 1);

        Mission &m = tops[0];
        m.blobIndex = m_tops[0];

        for (size_t i = 0; i < bottoms[0].h_data.size(); ++i)
        {
            Tensor::ptr bt = bottoms[0].h_data[i];
            std::uint32_t w = bt->width();
            std::uint32_t h = bt->height();

            Tensor::ptr t(new Tensor(w, h, sizeof(double), m_cpu_allocator, m_gpu_allocator));

            IntegralRowKernel<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>((const double *)bt->gpu_data(), w, h, (double *)t->mutable_gpu_data());
            IntegralColumnKernel<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>((double *)t->mutable_gpu_data(), w, h);

            // IntegralRowKernelS1<<<CUDA_GET_BLOCKS(t->count() / 4), CUDA_NUM_THREADS, 0, m_stream>>>(t->count() / 4, (const double *)bt->gpu_data(), (double *)t->mutable_gpu_data());
            // IntegralRowKernelS2<<<CUDA_GET_BLOCKS(h), CUDA_NUM_THREADS, 0, m_stream>>>((double *)t->mutable_gpu_data(), w, h);
            // IntegralRowKernelS3<<<CUDA_GET_BLOCKS(t->count() / 4), CUDA_NUM_THREADS, 0, m_stream>>>(t->count() / 4, (double *)t->mutable_gpu_data(), w);

            // IntegralColumnKernelS1<<<CUDA_GET_BLOCKS(t->count() / 4), CUDA_NUM_THREADS, 0, m_stream>>>(t->count() / 4, (double *)t->mutable_gpu_data(), w);
            // IntegralColumnKernelS2<<<CUDA_GET_BLOCKS(w), CUDA_NUM_THREADS, 0, m_stream>>>((double *)t->mutable_gpu_data(), w, h);
            // IntegralColumnKernelS3<<<CUDA_GET_BLOCKS(t->count() / 4), CUDA_NUM_THREADS, 0, m_stream>>>(t->count() / 4, (double *)t->mutable_gpu_data(), w);

            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime