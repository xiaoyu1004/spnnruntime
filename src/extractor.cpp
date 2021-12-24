#include "extractor.h"
#include "mission.h"
#include "setting/params.h"
#include "tensor.h"

#include <list>
#include <condition_variable>
#include <mutex>

namespace spnnruntime
{
    class Extractor::impl
    {
    public:
        impl() : m_flow(0), m_wave_flow(0)
        {
        }

    public:
        void pushMission(Mission &m)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_missions.push_back(m);
            m_condition.notify_one();
        }

        void extract(ToDataProc *proc)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_condition.wait(lock, [this](){ return !m_missions.empty(); });
            Mission m = m_missions.front();
            m_missions.pop_front();
            lock.unlock();
            parse(m, proc);
        }

    private:
        void parse(Mission &m, ToDataProc *proc)
        {
            if (m.datatype == DataType::report)
            {
                std::copy((const unsigned char*)m.report_data->cpu_data(), (const unsigned char*)m.report_data->cpu_data() + g_reportLen, proc->bitInfo);
                proc->bIsReport = true;
                m_wave_flow = 0;
                SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "hui gao information!";
                return;
            }

            unsigned char *head = (unsigned char *)m.head_data->cpu_data();
            std::uint32_t trgNum = 0;
            for (size_t i = 0; i < m.h_data.size(); ++i)
            {
                Tensor::ptr t = m.h_data[i];
                std::uint32_t w = t->width();
                std::uint32_t num = t->height();

                for (std::uint32_t k = 0; k < num; ++k)
                {
                    proc->pciSpPlotPacket.plots[trgNum + k].flow = m_flow;
                    proc->pciSpPlotPacket.plots[trgNum + k].azimuth = (short int)(head[16] + head[17] * 256) / 10.f;
                    proc->pciSpPlotPacket.plots[trgNum + k].elevation = (short int)(head[18] + head[19] * 256) / 10.f;
                    proc->pciSpPlotPacket.plots[trgNum + k].range = (w == 4096 ? 3000 : w == 2048 ? 1800: 0) + ((double *)t->row(k))[0] * 6;
                    proc->pciSpPlotPacket.plots[trgNum + k].velocity = m.intertype == 2 ? (128 - ((double *)t->row(k))[1]) * 0.7324 : (128 - ((double *)t->row(k))[1]) * 0.3256;
                    proc->pciSpPlotPacket.plots[trgNum + k].amplitude = ((double *)t->row(k))[3];
                    m_flow++;
                }
                trgNum += num;
            }
            proc->pciSpPlotPacket.ctrl.flow = m_wave_flow;
            proc->pciSpPlotPacket.ctrl.plotCnt = trgNum;
            proc->pciSpPlotPacket.ctrl.tgtNum = trgNum;
            proc->pciSpPlotPacket.ctrl.initTimeSlice = static_cast<int>(time(NULL));
            proc->pciSpPlotPacket.ctrl.azimuth = (short int)(head[16] + head[17] * 256) / 10.f;
            proc->pciSpPlotPacket.ctrl.elevation = (short int)(head[18] + head[19] * 256) / 10.f;
            proc->bIsReport = false;
            m_wave_flow++;
        }

    private:
        std::list<Mission> m_missions;
        std::condition_variable m_condition;
        std::mutex m_mutex;
        std::uint32_t m_flow;
        std::uint32_t m_wave_flow;
    };

    Extractor::Extractor() : m_impl(new impl)
    {
    }

    Extractor::~Extractor()
    {
        delete m_impl;
    }

    void Extractor::pushMission(Mission &m)
    {
        m_impl->pushMission(m);
    }

    void Extractor::extract(ToDataProc *proc)
    {
        m_impl->extract(proc);
    }
} // spnnruntime