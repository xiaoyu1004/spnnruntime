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
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime