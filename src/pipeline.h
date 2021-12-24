#ifndef PIPELINE_H
#define PIPELINE_H

#include "SPNNDefine.h"
#include "utils/common.h"
#include "tensor.h"
#include "extractor.h"
#include "blob.h"
#include "layer.h"

#include <string>
#include <memory>
#include <unordered_map>

namespace spnnruntime
{
    class SPNN_EXPORT pipeline
    {
    public:
        friend class Extractor;
        typedef std::shared_ptr<pipeline> ptr;
        class impl;

        pipeline();
        ~pipeline();

    public:
        void loadParam(std::string &&);
        void loadModel(std::string &&);
        void load_param_bin(std::string &&protopath);

        // lanuch all thread
        void start();
        // stop layer task
        void stop();

        // send cmd
        bool sendLaunchCmd(const unsigned char *_ptr);
        bool sendControlCmd(const unsigned char *_ptr);

        Extractor::ptr getExtractor() const;

        void enableProfiler();

        void disableProfiler();

    private:
        DISABLE_COPY_AND_ASSIGN(pipeline)
        impl *m_impl;
    };
}

#endif // spnnruntime