#include "hamming.h"
#include "utils/winfunc.hpp"

namespace spnnruntime
{
    // load layer specific parameter from parsed dict
    void Hamming::loadParam(const ParamDict::ptr pd)
    {
        m_hamming_data = hamming<double>(g_pulseNumber, 1.0);
        m_hamming_data->gpu_data();
    }

    DEFINE_LAYER_CREATOR(Hamming)
} // spnnruntime