#ifndef LAYER_H
#define LAYER_H

#include "SPNNDefine.h"
#include "utils/common.h"
#include "tensor.h"
#include "paramdict.h"
#include "modelbin.h"
#include "option.h"
#include "mission.h"
#include "extractor.h"
#include "log.h"
#include "blob.h"
#include "gpu.h"
#include "profiler.hpp"

#include <vector>
#include <string>
#include <functional>
#include <list>
#include <mutex>
#include <condition_variable>
#include <cstring>

#define DEFINE_LAYER_CREATOR(name)                                \
    ::spnnruntime::Layer *name##LayerCreator(void * /*userdata*/) \
    {                                                             \
        return new name;                                          \
    }

#define DEFINE_LAYER_DESTROYER(name)                                            \
    void name##LayerDestroyer(::spnnruntime::Layer *layer, void * /*userdata*/) \
    {                                                                           \
        delete layer;                                                           \
    }

namespace spnnruntime
{
    class pipeline;
    class Layer
    {
    public:
        using ptr = std::shared_ptr<Layer>;
        // empty
        Layer() : m_terminate(false)
        {
            CUDA_CHECK(cudaStreamCreate(&m_stream));
            m_cpu_allocator = new PoolAllocator(Device::CPU);
            m_gpu_allocator = new PoolAllocator(Device::GPU);
            m_gpu_allocator->stream(&m_stream);
        }

        // virtual destructor
        virtual ~Layer()
        {
            delete m_cpu_allocator;
            delete m_gpu_allocator;
        }

    public:
        // const cudaStream_t &stream() const { return m_stream; }
        
    public:
        // load layer specific parameter from parsed dict
        virtual void loadParam(const ParamDict::ptr pd);

        // load layer specific weight data from model binary
        virtual void loadModel(const ModelBin::ptr mb);

        virtual void forward(const std::vector<Mission> &bottoms, std::vector<Mission> &tops, const Option &opt)
        {
            /* do nothing */
        }

    private:
        friend class pipeline;
        std::vector<Mission> getMission();
        void pushMission(Mission &m);

        void stop() { m_terminate = true; }
        void notify() { m_condition.notify_one(); }
        size_t mcount() const { return m_missions.size(); }
        void streamsync() { CUDA_CHECK(cudaStreamSynchronize(m_stream)); }
        bool isTerminate() const { return m_terminate; }
        profiler::ptr getProfiler() { return m_prof; }
        void initProfiler() { m_prof.reset(new profiler); }

    public:
        // layer type index
        int m_typeindex;
        // layer type name
        std::string m_type;
        // layer name
        std::string m_name;
        // blob index which this layer needs as input
        std::vector<int> m_bottoms;
        // blob index which this layer produces as output
        std::vector<int> m_tops;

    protected:
        // allocator
        Allocator *m_cpu_allocator;
        Allocator *m_gpu_allocator;
        // task queue
        std::list<Mission> m_missions;
        // stream
        cudaStream_t m_stream;
        // mutex
        std::mutex m_mutex;
        std::condition_variable m_condition;
        bool m_terminate;

    private:
        DISABLE_COPY_AND_ASSIGN(Layer)
        profiler::ptr m_prof;
    };

    // get layer type from type name
    int LayerToIndex(const char *type);
    // create layer from type name
    Layer::ptr CreateLayer(const char *type);
    // create layer from layer type
    Layer::ptr CreateLayer(int index);

    // layer factory function
    typedef std::function<Layer *(void *)> LayerCreatorFunc;
    typedef std::function<void(Layer *, void *)> LayerDestroyerFunc;

    struct LayerRegistryEntry
    {
        // layer type name
        const char *name;
        // layer factory entry
        LayerCreatorFunc creator;
    };

    struct CustomLayerRegistryEntry
    {
        // layer type name
        const char *name;
        // layer factory entry
        LayerCreatorFunc creator;
        LayerDestroyerFunc destroyer;
        void *userdata;
    };
}
#endif