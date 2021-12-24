#ifndef PARAMS_H
#define PARAMS_H

#include <map>
#include <vector>

namespace spnnruntime
{
    extern std::uint32_t g_pulseNumber;
    extern std::uint32_t g_headLen;
    extern std::uint32_t g_reportLen;
    extern std::vector<std::uint32_t> g_pulse_lens;
    extern std::map<std::uint32_t, std::vector<std::uint32_t>> g_waves;
} // spnnruntime

#endif