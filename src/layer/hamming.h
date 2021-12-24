#ifndef HAMMING_H
#define HAMMING_H

#include "layer.h"
#include "setting/params.h"
#include "setting/radar.h"

namespace spnnruntime
{
    class Hamming : public Layer
    {
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;

    private:
        Tensor::ptr m_hamming_data;
    };
} // namespace spnnruntime

#endif