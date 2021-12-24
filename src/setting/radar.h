#ifndef RADAR_H
#define RADAR_H

#include <map>

namespace spnnruntime
{
    enum DataType {wave, report};
    
    struct ClusterInfo
    {
        double distance;
        double speed;
        double num;
        double amplitude;
    };

    class RadarParams
    {
    public:
        static void update()
        {
        }

    public:
        // cfar
        static double PFA_4096_1024;
        static double PFA_1024;
        static std::uint32_t referLen;
        static std::uint32_t guardLen;

        // cluster
        static double clstrRadius;
        static double clstrAreaThresh;
        static double imgRngRes;
        static double imgAzmRes;
    };

    struct PciPlot
    {
        unsigned int PlotHead : 16;  
        unsigned int flow : 8;       
        float azimuth;               
        float elevation;             
        unsigned int range : 16;     
        unsigned int amplitude : 16; 
        float velocity;              
        unsigned int BAK;
        unsigned int BAKxx;
    };

    struct PciCtrlBit
    {
        unsigned int Head : 16;         
        unsigned int flow : 24;         
        unsigned int plotCnt : 8;       
        unsigned int tgtNum : 16;       
        unsigned int initTimeSlice : 4; 
        unsigned int jobType : 4;       
        unsigned int waveNum : 8;       
        float azimuth;                  
        float elevation;                
        unsigned int Sum : 8;
        int deltaA : 8;
        int deltaE : 8;
        unsigned int assist : 8;
        unsigned int SNR_S : 8;
        unsigned int SNR_N : 8;
        unsigned int BITInfo : 8;
        unsigned int JamInfo : 8;
        unsigned int BAK1;
        unsigned int BAK2 : 16;
    };

    struct PciSpPlotPacket
    {
        PciCtrlBit ctrl;
        PciPlot plots[30];
    };

    struct ToDataProc
    {
        PciSpPlotPacket pciSpPlotPacket;
        unsigned char bitInfo[128];
        bool bIsReport;
    };
} // spnnruntime

#endif