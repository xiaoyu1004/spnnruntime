#ifndef MISSION_H
#define MISSION_H

#include "tensor.h"
#include "setting/radar.h"

namespace spnnruntime
{
    class Mission
    {
    public:
        Mission() 
        {
        }

        bool operator==(const Mission &m)
        {
            return m.mid == mid && m.intertype == intertype && m.blobIndex == blobIndex;
        }

    public:
        std::uint64_t mid;
        std::uint32_t blobIndex;
        std::uint32_t intertype;
        DataType datatype;

        Tensor::ptr head_data;
        Tensor::ptr report_data;
        std::vector<Tensor::ptr> h_data;
        std::vector<Tensor::ptr> c_data;
    };
} // spnnruntime

#endif