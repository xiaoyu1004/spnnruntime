#ifndef INPUT_H
#define INPUT_H

#include "layer.h"
#include "setting/params.h"
#include "setting/radar.h"
#include "command.h"

namespace spnnruntime
{
    class Input : public Layer
    {
    public:
        Input();
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd) override;

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt) override;

    private:
        std::uint64_t m_missionId;
        Command::ptr m_cmd;
    };
} // spnnruntime

#endif