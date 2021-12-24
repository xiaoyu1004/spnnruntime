#include "layer.h"

namespace spnnruntime
{
    class AbsVal : public Layer
    {
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;
    };
} // spnnruntime