#ifndef PULSECOMPRESS_H
#define PULSECOMPRESS_H

#include "layer.h"
#include "gpu.h"

#include <cufft.h>

#include <vector>
#include <map>

namespace spnnruntime
{
    class PulseCompress : public Layer
    {
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        // load layer specific weight data from model binary
        virtual void loadModel(const ModelBin::ptr mb) override;

        void initPcFFT();
        
        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;

    private:
        std::map<std::uint32_t, Tensor::ptr> m_coefs;
        std::map<std::uint32_t, cufftHandle> m_pc_fft_handles;
    };
} // spnnruntime

#endif