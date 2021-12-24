#ifndef BINARYOP_H
#define BINARYOP_H

#include "layer.h"

namespace spnnruntime
{
    class BinaryOp : public Layer
    {
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;

    private:
        enum OpType {plus, minus, product, divide};
        OpType m_optype;
    };
} // spnnruntime

#endif