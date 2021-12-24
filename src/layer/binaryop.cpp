#include "binaryop.h"

namespace spnnruntime
{
    // load layer specific parameter from parsed dict
    void BinaryOp::loadParam(const ParamDict::ptr pd)
    {
        m_optype = (OpType)pd->get(0, 0);
    }

    DEFINE_LAYER_CREATOR(BinaryOp)
} // spnnruntime