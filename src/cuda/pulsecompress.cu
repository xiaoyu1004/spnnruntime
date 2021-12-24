#include "layer/pulsecompress.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void PointMulScale(const std::uint32_t N, ComplexType *in, const ComplexType *coef, std::uint32_t w)
    {
        double scale = 1.0 / w;
        std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            in[tid] = cuCmul(in[tid], coef[tid % w]);
            in[tid].x *= scale;
            in[tid].y *= scale;
        }
    }

    void PulseCompress::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
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

            Tensor::ptr t(new Tensor(w, h, sizeof(ComplexType), m_cpu_allocator, m_gpu_allocator));
            CUFFT_CHECK(cufftExecZ2Z(m_pc_fft_handles[w], (ComplexType *)bt->gpu_data(), (ComplexType *)t->mutable_gpu_data(), CUFFT_FORWARD));
            PointMulScale<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>(t->count(), (ComplexType *)t->mutable_gpu_data(), (const ComplexType *)m_coefs[w]->gpu_data(), w);
            CUFFT_CHECK(cufftExecZ2Z(m_pc_fft_handles[w], (ComplexType *)t->gpu_data(), (ComplexType *)t->mutable_gpu_data(), CUFFT_INVERSE));
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime