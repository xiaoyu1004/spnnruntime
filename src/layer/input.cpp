#include "input.h"

namespace spnnruntime
{
    Input::Input() : m_missionId(0), m_cmd(CmdMgr::GetInstance())
    {
    }

    void Input::loadParam(const ParamDict::ptr pd)
    {
    }

    DEFINE_LAYER_CREATOR(Input)
} // spnnruntime