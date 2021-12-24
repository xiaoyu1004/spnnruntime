#include "clusterop.h"
#include "setting/params.h"
#include "setting/radar.h"
#include "cpu.h"

namespace spnnruntime
{
    // load layer specific parameter from parsed dict
    void ClusterOp::loadParam(const ParamDict::ptr pd)
    {
    }

    void ClusterOp::forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
    {
        CHECK_EQ(bottoms.size(), 1);
        CHECK_EQ(tops.size(), 1);

        Mission &m = tops[0];
        m.blobIndex = m_tops[0];
        /*
        for (size_t i = 0; i < bottoms[0].h_data.size(); ++i)
        {
            Tensor::ptr bt = bottoms[0].h_data[i];
            std::uint32_t w = bt->width();
            std::uint32_t h = bt->height();
            Tensor::ptr mine_pos(new Tensor(w * 3, h, sizeof(double), m_cpu_allocator, m_gpu_allocator));

            double ClstrRadius = RadarParams::clstrRadius;        
            double ImgRngRes = RadarParams::imgRngRes;            
            double ImgAzmRes = RadarParams::imgAzmRes;            
            double ClstrAreaThresh = RadarParams::clstrAreaThresh;
            std::uint32_t TempTrgNum, IsIsltPnt;
            std::uint32_t TempInt1, TempInt2;
            double TempfR1, TempfR2, TempfC1, TempfC2;
            double ClstrRadEdg, CntDstnc;
            std::uint32_t rr, cc;
            double TempFloat1, TempFloat2;

            mine_pos->fill(0.0);
            double *mine_data = (double *)mine_pos->cpu_data();
            m_clstr_vec.clear();
            m_clstr_vec.reserve(100);
            if (ImgAzmRes > ImgRngRes)
            {
                ClstrRadEdg = std::ceil(ClstrRadius / ImgRngRes) + 1;
            }
            else
            {
                ClstrRadEdg = std::ceil(ClstrRadius / ImgAzmRes) + 1;
            }
            std::uint32_t TrgtAreaNum = std::ceil(0.33 / ImgAzmRes) * std::ceil(0.33 / ImgRngRes) * ClstrAreaThresh;

            TempInt1 = 0;
            const double *bt_data = (const double *)bt->cpu_data();
#ifdef _OPENMP
#pragma omp parallel for num_threads(12)
#endif
            for (rr = 0; rr < h; rr++)
            {
                for (cc = 0; cc < w; cc++)
                {
                    if (bt_data[rr * w + cc] > 0)
                    {
                        mine_data[TempInt1 * 3] = cc;
                        mine_data[TempInt1 * 3 + 1] = rr;
                        mine_data[TempInt1 * 3 + 2] = bt_data[rr * w + cc];
                        TempInt1++;
                    }
                }
            }
            if (TempInt1 == 0)
            {
                continue;
            }

            TempTrgNum = 1;
            ClusterInfo info;
            info.distance = mine_data[0];
            info.speed = mine_data[1];
            info.amplitude = mine_data[2];
            info.num = 1.0;
            m_clstr_vec.push_back(info);
            for (rr = 1; rr < TempInt1; rr++)
            {
                IsIsltPnt = 1;
                TempfR1 = mine_data[rr * 3];
                TempfC1 = mine_data[rr * 3 + 1];
                TempFloat1 = mine_data[rr * 3 + 2];
                for (cc = 0; cc < TempTrgNum; cc++)
                {
                    TempfR2 = m_clstr_vec[cc].distance;
                    TempfC2 = m_clstr_vec[cc].speed;
                    TempFloat2 = m_clstr_vec[cc].amplitude;
                    TempInt2 = m_clstr_vec[cc].num;
                    CntDstnc = std::abs(TempfR2 - TempfR1);
                    if (CntDstnc <= ClstrRadEdg)
                    {
                        TempfC2 = (TempfC1 * TempFloat1 + TempfC2 * TempFloat2) / (TempFloat2 + TempFloat1);
                        if (TempFloat1 > TempFloat2)
                        {
                            m_clstr_vec[cc].distance = TempfR1;
                        }
                        m_clstr_vec[cc].speed = (TempfC1 + TempfC2) / 2;
                        m_clstr_vec[cc].amplitude = std::max(TempFloat2, TempFloat1);
                        m_clstr_vec[cc].num = TempInt2 + 1;
                        IsIsltPnt = 0;
                        break;
                    }
                }
                if (IsIsltPnt == 1)
                {
                    ClusterInfo info;
                    info.distance = TempfR1;
                    info.speed = TempfC1;
                    info.amplitude = TempFloat1;
                    info.num = 1;
                    m_clstr_vec.push_back(info);
                    TempTrgNum++;
                }
            }

            std::vector<ClusterInfo> result;
            for (rr = 0; rr < TempTrgNum; rr++)
            {
                int speed = (int)(m_clstr_vec[rr].speed + 0.5);
                std::uint32_t num = m_clstr_vec[rr].num;
                if (num >= TrgtAreaNum && (speed > 131 || speed < 125))
                {
                    result.push_back(m_clstr_vec[rr]);
                }
            }

            if (result.size() > 0)
            {
                Tensor::ptr t(new Tensor(4, result.size(), sizeof(double), m_cpu_allocator, m_gpu_allocator));
                double *top_data = (double *)t->mutable_cpu_data();
                for (size_t i = 0; i < result.size(); ++i)
                {
                    top_data[i * 4] = result[i].distance;
                    top_data[i * 4 + 1] = result[i].speed;
                    top_data[i * 4 + 2] = result[i].num;
                    top_data[i * 4 + 3] = result[i].amplitude;
                }
                m.h_data.push_back(t);
            }
        }
        */
        // SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "layer name = " << m_name << " mission count = " << m_missions.size() << " mid = " << m.mid;
    }

    DEFINE_LAYER_CREATOR(ClusterOp)
} // spnnruntime