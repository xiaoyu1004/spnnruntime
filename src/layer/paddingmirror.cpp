#include "paddingmirror.h"

namespace spnnruntime
{
    // load layer specific parameter from parsed dict
    void PaddingMirror::loadParam(const ParamDict::ptr pd)
    {
    }

    DEFINE_LAYER_CREATOR(PaddingMirror)
} // spnnruntime