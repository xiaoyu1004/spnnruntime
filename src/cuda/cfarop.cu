#include "layer/cfarop.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    __inline__ __device__ double BoxIntegral(const double *data, std::uint32_t width, std::uint32_t r1, std::uint32_t r2, std::uint32_t c1, std::uint32_t c2)
    {
        return data[(r1) * (width) + (c1)] +
               data[(r2) * (width) + (c2)] -
               data[(r1) * (width) + (c2)] -
               data[(r2) * (width) + (c1)];
    }

    __global__ void CfarKernel(const std::uint32_t N, const double *in, const std::uint32_t n, std::uint32_t w, const double *integral_image, double alpha, std::uint32_t referLength, std::uint32_t guardLength, unsigned char *out)
    {
        std::uint32_t index = GET_THREAD_ID();
        double s1, s2;
        double mean, t;
        const double EPS = 1e-8f;
        std::uint32_t padded_width = w + 2 * referLength;
        std::uint32_t c = index % w, r = index / w;

        if (index < N)
        {
            s1 = BoxIntegral(integral_image, padded_width,
                              r, r + 2 * referLength,
                              c, c + 2 * referLength);

            s2 = BoxIntegral(integral_image, padded_width,
                              r + referLength - guardLength, r + referLength + guardLength,
                              c + referLength - guardLength, c + referLength + guardLength);
            mean = (s1 - s2) / n;
            t = alpha * mean;
            out[index] = (in[index] - t > EPS) ? 1 : 0;
        }
    }

    void CfarOp::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
    {
        CHECK_EQ(bottoms.size(), 2);
        CHECK_EQ(tops.size(), 1);

        Mission &m = tops[0];
        m.blobIndex = m_tops[0];

        std::uint32_t n = (4 * RadarParams::referLen * RadarParams::referLen) - (4 * RadarParams::guardLen * RadarParams::guardLen);
        for (size_t i = 0; i < bottoms[0].h_data.size(); ++i)
        {
            Tensor::ptr bt = bottoms[0].h_data[i];
            Tensor::ptr integral_image = bottoms[1].h_data[i];
            std::uint32_t w = bt->width();
            std::uint32_t h = bt->height();

            Tensor::ptr t(new Tensor(w, h, sizeof(unsigned char), m_cpu_allocator, m_gpu_allocator));
            double alpha = n * (powf(w == 1024 ? RadarParams::PFA_1024 : RadarParams::PFA_4096_1024, -1.0f / n) - 1);
            CfarKernel<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>(t->count(), (const double *)bt->gpu_data(), n, w, (const double *)integral_image->gpu_data(), alpha, RadarParams::referLen, RadarParams::guardLen, (unsigned char *)t->mutable_gpu_data());
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime