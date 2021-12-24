#ifndef MOVEDETECTOR_H
#define MOVEDETECTOR_H

#include "layer.h"

#include <cufft.h>

namespace spnnruntime
{
    class MoveDetector : public Layer
    {
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        void initMtdFFT();

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;

    private:
        std::vector<std::uint32_t> m_pulse_lens;
        std::map<std::uint32_t, cufftHandle> m_mtd_fft_handles;
    };
} // spnnruntime

#endif