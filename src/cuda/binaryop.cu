#include "layer/binaryop.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void MaskKernel(const std::uint32_t N, const unsigned char *bool_data, const double *mtd_data, double *mask_data)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            mask_data[tid] = static_cast<double>(bool_data[tid]) * mtd_data[tid];
        }
    }

    void BinaryOp::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
    {
        CHECK_EQ(bottoms.size(), 2);
        CHECK_EQ(tops.size(), 1);

        Mission &m = tops[0];
        m.blobIndex = m_tops[0];

        for (size_t i = 0; i < bottoms[0].h_data.size(); ++i)
        {
            Tensor::ptr mtd_data = bottoms[0].h_data[i];
            Tensor::ptr cfar_data = bottoms[1].h_data[i];
            std::uint32_t w = cfar_data->width();
            std::uint32_t h = cfar_data->height();

            Tensor::ptr t(new Tensor(w, h, sizeof(double), m_cpu_allocator, m_gpu_allocator));
            if (m_optype == OpType::product)
            {
                MaskKernel<<<CUDA_GET_BLOCKS(cfar_data->count()), CUDA_NUM_THREADS, 0, m_stream>>>(cfar_data->count(), (const unsigned char *)cfar_data->gpu_data(), (const double *)mtd_data->gpu_data(), (double *)t->mutable_gpu_data());
                t->cpu_data();
            }
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime