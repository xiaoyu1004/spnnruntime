#ifndef INTEGRAL_H
#define INTEGRAL_H

#include "layer.h"

namespace spnnruntime
{
    class Integral : public Layer
    {
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;
    };
} // spnnruntime

#endif