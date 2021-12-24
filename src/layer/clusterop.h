#ifndef CLUSTEROP_H
#define CLUSTEROP_H

#include "layer.h"
#include "setting/radar.h"

namespace spnnruntime
{
    class ClusterOp : public Layer
    {
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;

    private:
        std::vector<ClusterInfo> m_clstr_vec;
    };
} // spnnruntime

#endif