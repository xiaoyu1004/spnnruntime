#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include "SPNNDefine.h"
#include "utils/common.h"
#include "mission.h"
#include "setting/radar.h"

#include <memory>

namespace spnnruntime
{
    class pipeline;
    class Layer;
    class SPNN_EXPORT Extractor
    {
    public:
        typedef std::shared_ptr<Extractor> ptr;
        class impl;
        ~Extractor();
        
    public:
        void extract(ToDataProc *toDataProc);

    protected:
        friend class pipeline;
        Extractor();

        friend class Layer;
        void pushMission(Mission &m);

    private:
        impl *m_impl;
    };
} // spnnruntime

#endif