#include "layer/paddingmirror.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    __global__ void CopyDataKernel(const std::uint32_t N, const double *in, const std::uint32_t s_w, const std::uint32_t d_w, const std::uint32_t referLength, double *out)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            out[(tid / s_w + referLength) * d_w + referLength + tid % s_w] = in[tid];
        }
    }

    __global__ void MirrorLeftKernel(const std::uint32_t N, double *in, std::uint32_t w, std::uint32_t referLength)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            in[tid / referLength * w + tid % referLength] = in[tid / referLength * w + 2 * referLength - tid % referLength - 1];
        }
    }

    __global__ void MirrorRightKernel(const std::uint32_t N, double *in, std::uint32_t w, std::uint32_t referLength)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            in[tid / referLength * w + w - 1 - tid % referLength] = in[tid / referLength * w + w - 2 * referLength + tid % referLength];
        }
    }

    __global__ void MirrorTopKernel(const std::uint32_t N, double *in, std::uint32_t w, std::uint32_t referLength)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            in[tid] = in[(2 * referLength - 1 - tid / w) * w + tid % w];
        }
    }

    __global__ void MirrorBottomKernel(const std::uint32_t N, double *in, std::uint32_t w, std::uint32_t h, std::uint32_t referLength)
    {
        const std::uint32_t tid = GET_THREAD_ID();
        if (tid < N)
        {
            in[(h - 1 - tid / w) * w + tid % w] = in[(h - 2 * referLength) * w + tid];
        }
    }

    void PaddingMirror::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
    {
        CHECK_EQ(bottoms.size(), 1);
        CHECK_EQ(tops.size(), 1);

        Mission &m = tops[0];
        m.blobIndex = m_tops[0];

        for (size_t i = 0; i < bottoms[0].h_data.size(); ++i)
        {
            Tensor::ptr bt = bottoms[0].h_data[i];
            std::uint32_t w = bt->width() + 2 * RadarParams::referLen;
            std::uint32_t h = bt->height() + 2 * RadarParams::referLen;

            const int LR_N = RadarParams::referLen * h;
            const int TB_N = RadarParams::referLen * w;

            Tensor::ptr t(new Tensor(w, h, sizeof(double), m_cpu_allocator, m_gpu_allocator));
            CopyDataKernel<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>(t->count(), (const double *)bt->gpu_data(), bt->width(), w, RadarParams::referLen, (double *)t->mutable_gpu_data());

            MirrorLeftKernel<<<CUDA_GET_BLOCKS(LR_N), CUDA_NUM_THREADS, 0, m_stream>>>(LR_N, (double *)t->mutable_gpu_data(), w, RadarParams::referLen);
            MirrorRightKernel<<<CUDA_GET_BLOCKS(LR_N), CUDA_NUM_THREADS, 0, m_stream>>>(LR_N, (double *)t->mutable_gpu_data(), w, RadarParams::referLen);
            MirrorTopKernel<<<CUDA_GET_BLOCKS(TB_N), CUDA_NUM_THREADS, 0, m_stream>>>(TB_N, (double *)t->mutable_gpu_data(), w, RadarParams::referLen);
            MirrorBottomKernel<<<CUDA_GET_BLOCKS(TB_N), CUDA_NUM_THREADS, 0, m_stream>>>(TB_N, (double *)t->mutable_gpu_data(), w, h, RadarParams::referLen);
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime