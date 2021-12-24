#include "layer/absval.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void AbsKernel(const std::uint32_t N, const ComplexType *in, double *out)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            out[tid] = cuCabs(in[tid]);
        }
    }

    void AbsVal::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
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
            AbsKernel<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>(t->count(), (const ComplexType *)bt->gpu_data(), (double *)t->mutable_gpu_data());
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime