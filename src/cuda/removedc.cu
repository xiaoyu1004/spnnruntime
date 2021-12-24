#include "layer/removedc.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void SumKernel(const std::uint32_t N, const ComplexType *in, ComplexType *sum, const std::uint32_t h)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            ComplexType tmp;
            tmp.x = 0.f;
            tmp.y = 0.f;
            for (std::uint32_t i = 0; i < h; ++i)
            {
                tmp = cuCadd(tmp, in[i * N + tid]);
            }
            tmp.x /= h;
            tmp.y /= h;
            sum[tid] = tmp;
        }
    }

    __global__ void ReduceMeanKernel(const std::uint32_t N, const ComplexType *in, const ComplexType *sum, ComplexType *out, const std::uint32_t w)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            out[tid] = cuCsub(in[tid], sum[tid % w]);
        }
    }

    void RemoveDC::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
    {
        CHECK_EQ(bottoms.size(), 1);
        CHECK_EQ(tops.size(), 1);

        Mission &m = tops[0];
        m.blobIndex = m_tops[0];
        for (size_t i = 0; i < bottoms[0].h_data.size(); ++i)
        {
            std::uint32_t w = bottoms[0].h_data[i]->width();
            std::uint32_t h = bottoms[0].h_data[i]->height();
            
            Tensor::ptr sum(new Tensor(w, sizeof(ComplexType), m_cpu_allocator, m_gpu_allocator));
            Tensor::ptr t(new Tensor(w, h, sizeof(ComplexType), m_cpu_allocator, m_gpu_allocator));

            SumKernel<<<CUDA_GET_BLOCKS(w), CUDA_NUM_THREADS, 0, m_stream>>>(w, (ComplexType *)bottoms[0].h_data[i]->gpu_data(), (ComplexType *)sum->mutable_gpu_data(), h);
            ReduceMeanKernel<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>(t->count(), (ComplexType *)bottoms[0].h_data[i]->gpu_data(), (ComplexType *)sum->gpu_data(), (ComplexType *)t->mutable_gpu_data(), w);
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime