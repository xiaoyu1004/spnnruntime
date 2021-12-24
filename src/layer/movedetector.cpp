#include "movedetector.h"
#include "setting/params.h"
#include "setting/radar.h"

namespace spnnruntime
{
    // load layer specific parameter from parsed dict
    void MoveDetector::loadParam(const ParamDict::ptr pd)
    {   
        initMtdFFT();
    }

    void MoveDetector::initMtdFFT()
    {
        for (size_t i = 0; i < g_pulse_lens.size(); ++i)
        {
            const int rank = 1;
            int n = (int)g_pulseNumber;

            int inembed[2];
            int onembed[2];

            inembed[0] = g_pulseNumber;
            inembed[1] = g_pulse_lens[i];
            int istride = g_pulse_lens[i];
            int idist = 1;

            onembed[0] = g_pulseNumber;
            onembed[1] = g_pulse_lens[i];
            int ostride = g_pulse_lens[i];
            int odist = 1;

            int batch = g_pulse_lens[i];

            cufftHandle mtdFFTHandle;
            CUFFT_CHECK(cufftPlanMany(&mtdFFTHandle, rank, &n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch));
            CUFFT_CHECK(cufftSetStream(mtdFFTHandle, m_stream));
            m_mtd_fft_handles[g_pulse_lens[i]] = mtdFFTHandle;
        }
    }

    DEFINE_LAYER_CREATOR(MoveDetector)
} // spnnruntime