#include "layer/hamming.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void HammingKernel(const std::uint32_t N, const ComplexType *in, const double *hamming_data, ComplexType *out, const std::uint32_t w)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            out[tid].x = in[tid].x * hamming_data[tid / w];
            out[tid].y = in[tid].y * hamming_data[tid / w];
        }
    }

    void Hamming::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
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
            HammingKernel<<<CUDA_GET_BLOCKS(bt->count()), CUDA_NUM_THREADS, 0, m_stream>>>(bt->count(), (const ComplexType *)bt->gpu_data(), (const double *)m_hamming_data->gpu_data(), (ComplexType *)t->mutable_gpu_data(), w);
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime