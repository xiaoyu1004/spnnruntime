#include "layer/input.h"
#include "setting/radar.h"
#include "gpu.h"

namespace spnnruntime
{
    static void DataFix(unsigned char *inter_data, const std::uint32_t intertype)
    {
        if (intertype == 1 || intertype == 3)
        {
            std::copy(inter_data + 60, inter_data + 60 + (5120 - 1) * 12, inter_data + 60 + 12);
        }
        else if (intertype == 2)
        {
            for (std::uint32_t i = 0; i < g_reportLen; ++i)
            {
                memset(inter_data + i * 36928 + 60 + 1799 * 12, 0, 249 * 12);
            }
        }
    }

    __global__ void ParseHeDataKernel(const std::uint32_t N, ComplexType *out_data, const std::uint32_t step, const std::uint32_t width, const std::uint32_t height, const unsigned char *inter_data, const std::uint32_t offset)
    {
        const int tid = GET_THREAD_ID();
        if (tid < N)
        {
            const int index = tid / width * step + tid % width * 12 + 60 + offset * 12;

            out_data[tid].y = (short int)(inter_data[index] + inter_data[index + 1] * 256);
            out_data[tid].x = (short int)(inter_data[index + 2] + inter_data[index + 3] * 256);
        }
    }
    
    void Input::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
    {
        CHECK_EQ(tops.size(), 1);
        Mission &m = tops[0];
        m.mid = m_missionId++;
        m.blobIndex = m_tops[0];

        if (m.mid >= 160000)
        {
            m_terminate = true;
        }

        std::uint32_t intertype = m_cmd->getInterType();
        m.intertype = intertype;
        DataType datatype = m_cmd->getDataType(intertype);
        m.datatype = datatype;

        auto it = g_waves.find(intertype);
        if (it == g_waves.end())
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "intertype error! not exists in g_waves!";
        }
        std::vector<std::uint32_t> params = it->second;
        Tensor::ptr inter_data(new Tensor(params[0], g_pulseNumber, 1, m_cpu_allocator, m_gpu_allocator));
        m_cmd->loadData(inter_data->mutable_cpu_data(), params[0] * g_pulseNumber);
        // read head data
        m.head_data.reset(new Tensor(60u, 1, m_cpu_allocator, m_gpu_allocator));
        std::copy((unsigned char *)inter_data->cpu_data(), (unsigned char *)inter_data->cpu_data() + g_headLen, (unsigned char *)m.head_data->mutable_cpu_data());
        if (datatype == DataType::report)
        {
            m.report_data.reset(new Tensor(g_reportLen, 1, m_cpu_allocator, m_gpu_allocator));
            std::copy((const unsigned char *)inter_data->cpu_data(), (const unsigned char *)inter_data->cpu_data() + g_reportLen, (unsigned char *)m.report_data->mutable_cpu_data());
            return;
        }
        DataFix((unsigned char *)inter_data->mutable_cpu_data(), intertype);
        // parse he
        std::uint32_t offset = 0;
        std::uint32_t size = params.size();
        for (size_t i = 1; i < size; ++i)
        {
            Tensor::ptr t(new Tensor(params[i], g_pulseNumber, sizeof(ComplexType), m_cpu_allocator, m_gpu_allocator));
            ParseHeDataKernel<<<CUDA_GET_BLOCKS(t->count()), CUDA_NUM_THREADS, 0, m_stream>>>(t->count(), (ComplexType *)t->mutable_gpu_data(), params[0], params[i], g_pulseNumber, (const unsigned char *)inter_data->gpu_data(), offset);
            offset += params[i];
            m.h_data.push_back(t);
        }
        CUDA_POST_KERNEL_CHECK;
    }
} // spnnruntime