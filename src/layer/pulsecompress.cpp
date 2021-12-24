#include "pulsecompress.h"
#include "setting/params.h"
#include "setting/radar.h"

namespace spnnruntime
{
    // load layer specific parameter from parsed dict
    void PulseCompress::loadParam(const ParamDict::ptr pd)
    {
        initPcFFT();
    }

    // load layer specific weight data from model binary
    void PulseCompress::loadModel(const ModelBin::ptr mb)
    {
        for (size_t i = 0; i < g_pulse_lens.size(); ++i)
        {
            Tensor::ptr magic = mb->load(1, 1);
            if (static_cast<const float *>(magic->cpu_data())[0] != 7767517)
            {
                SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "load data error! layer name: " << m_name;
            }
            m_coefs[g_pulse_lens[i]] = mb->load(g_pulse_lens[i], 2);
        }
    }

    void PulseCompress::initPcFFT()
    {
        for (size_t i = 0; i < g_pulse_lens.size(); ++i)
        {
            const int rank = 1;
            int n = (int)g_pulse_lens[i];

            int inembed[2];
            int onembed[2];

            inembed[0] = g_pulse_lens[i];
            inembed[1] = g_pulseNumber;
            int istride = 1;
            int idist = g_pulse_lens[i];

            onembed[0] = g_pulse_lens[i];
            onembed[1] = g_pulseNumber;
            int ostride = 1;
            int odist = g_pulse_lens[i];

            int batch = g_pulseNumber;

            cufftHandle pcFFTHandle;
            CUFFT_CHECK(cufftPlanMany(&pcFFTHandle, rank, &n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch));
            CUFFT_CHECK(cufftSetStream(pcFFTHandle, m_stream));
            m_pc_fft_handles[g_pulse_lens[i]] = pcFFTHandle;
        }
    }

    DEFINE_LAYER_CREATOR(PulseCompress)
} // spnnruntime