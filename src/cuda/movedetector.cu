#include "layer/movedetector.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void FFTShiftKernel(const std::uint32_t N, const ComplexType *in, const std::uint32_t w, const std::uint32_t h, ComplexType *out)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            const std::uint32_t index = tid / w;
            const std::uint32_t limit = h / 2;
            if (index >= limit)
            {
                out[tid - N / 2] = in[tid];
            }
            else
            {
                out[tid + N / 2] = in[tid];
            }
        }
    }

    void MoveDetector::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
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

            Tensor::ptr fft(new Tensor(w, h, sizeof(ComplexType), m_cpu_allocator, m_gpu_allocator));
            Tensor::ptr shift(new Tensor(w, h, sizeof(ComplexType), m_cpu_allocator, m_gpu_allocator));

            CUFFT_CHECK(cufftExecZ2Z(m_mtd_fft_handles[w], (ComplexType *)bt->gpu_data(), (ComplexType *)fft->mutable_gpu_data(), CUFFT_FORWARD));
            FFTShiftKernel<<<CUDA_GET_BLOCKS(fft->count()), CUDA_NUM_THREADS, 0, m_stream>>>(fft->count(), (const ComplexType *)fft->gpu_data(), w, h, (ComplexType *)shift->mutable_gpu_data());
            m.h_data.push_back(shift);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime