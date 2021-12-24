#include "radar.h"

#include <map>
#include <vector>

namespace spnnruntime
{
    // pulse number
    std::uint32_t g_pulseNumber = 256;
    // report length
    std::uint32_t g_headLen = 60u;
    std::uint32_t g_reportLen = 128u;
    std::vector<std::uint32_t> g_pulse_lens = {4096, 2048, 1024};
    // intertype -> waves
    std::map<std::uint32_t, std::vector<std::uint32_t>> g_waves = {
        {1, {61504, 4096, 1024}},
        {2, {36928, 2048, 1024}},
        {3, {61504, 4096, 1024}},
        {4, {128}} // report
    };

    // cfar
    double RadarParams::PFA_4096_1024 = 5e-4f;
    double RadarParams::PFA_1024 = 5e-5f;
    std::uint32_t RadarParams::referLen = 15;
    std::uint32_t RadarParams::guardLen = 5;

    // cluster
    double RadarParams::clstrRadius = 0.25;
    double RadarParams::clstrAreaThresh = 0.04;
    double RadarParams::imgRngRes = 0.05;
    double RadarParams::imgAzmRes = 0.05;
} // spnnruntime