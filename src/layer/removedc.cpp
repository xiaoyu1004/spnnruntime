#include "removedc.h"

namespace spnnruntime
{
    // load layer specific parameter from parsed dict
    void RemoveDC::loadParam(const ParamDict::ptr pd)
    {
    }

    DEFINE_LAYER_CREATOR(RemoveDC)
} // spnnruntime