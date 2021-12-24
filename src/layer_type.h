#ifndef LAYER_TYPE_H
#define LAYER_TYPE_H

#include <cstdint>

namespace spnnruntime
{
    enum class LayerType : std::uint8_t
    {
#include "layer_type_enum.h"
        CustomBit = (1 << 8),
    };

} // namespace ncnn

#endif
