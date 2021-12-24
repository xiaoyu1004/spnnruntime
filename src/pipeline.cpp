#include "pipeline.h"
#include "datareader.h"
#include "log.h"
#include "command.h"
#include "profiler.hpp"
#include "utils/check.hpp"

#include <fstream>
#include <future>
#include <algorithm>
#include <cstdio>

namespace spnnruntime
{
    class pipeline::impl
    {
    public:
        typedef std::shared_ptr<impl> ptr;
        impl() : m_ex(new Extractor), m_cmd(CmdMgr::GetInstance()), m_profile(false)
        {
        }

        ~impl()
        {
            clear();
        }

    public:
        void loadParam(std::string &&parampath)
        {
            FILE *fp = fopen(parampath.c_str(), "rb");
            if (fp == NULL)
            {
                SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "open file [" << parampath << "] failed!";
            }
            loadParam(fp);
            fclose(fp);
        }

        void loadModel(std::string &&modelpath)
        {
            FILE *fp = fopen(modelpath.c_str(), "rb");
            if (!fp)
            {
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "fopen " << modelpath << " failed";
            }
            loadModel(fp);
            fclose(fp);
        }

        void load_param_bin(std::string &&protopath)
        {
            // TODO
        }

        void start()
        {
            if (!m_threads.empty())
            {
                return;
            }

            for (size_t i = 0; i < m_layers.size(); ++i)
            {
                Option opt;
                m_threads.push_back(std::make_shared<std::thread>(std::bind(&pipeline::impl::forward, this, std::placeholders::_1, std::placeholders::_2), m_layers[i], opt));
                // SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "start forward layer name: " << m_layers[i]->m_name;
            }
        }

        void wait()
        {
            for (auto it = m_threads.begin(), end = m_threads.end(); it != end; ++it)
            {
                if ((*it)->joinable())
                {
                    (*it)->join();
                }
            }
        }

        void stop()
        {
            for (size_t i = 0; i < m_layers.size(); ++i)
            {
                m_layers[i]->stop();
                m_layers[i]->notify();
            }
        }

        bool sendLaunchCmd(const unsigned char *_ptr)
        {
            return m_cmd->sendLaunchCmd(_ptr);
        }

        bool sendControlCmd(const unsigned char *_ptr)
        {
            return m_cmd->sendControlCmd(_ptr);
        }

        void enableProfiler()
        {
            m_profile = true;
            for (size_t i = 0; i < m_layers.size(); ++i)
            {
                m_layers[i]->initProfiler();
            }
        }

        void disableProfiler()
        {
            m_profile = false;
        }

    private:
        void loadParam(FILE *fp)
        {
            DataReaderFromStdio::ptr dr(new DataReaderFromStdio(fp));
            loadParam(dr);
        }

        void loadParam(const DataReader::ptr dr)
        {
#define SCAN_VALUE(fmt, v)                                                \
    if (!dr->scan(fmt, &v))                                               \
    {                                                                     \
        SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "parse " << #v << " failed!"; \
    }
            int magic = 0;
            SCAN_VALUE("%d", magic)
            if (magic != 7767517)
            {
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "param is error, please regenerate!";
            }
            int layerCount = 0;
            int blobCount = 0;
            SCAN_VALUE("%d", layerCount)
            SCAN_VALUE("%d", blobCount)
            if (layerCount <= 0 || blobCount <= 0)
            {
                SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "invalid layer_count or blob_count";
            }
            m_layers.resize(static_cast<size_t>(layerCount));
            m_blobs.resize(static_cast<size_t>(blobCount));

            ParamDict::ptr pd(new ParamDict);
            int blobIndex = 0;
            for (int i = 0; i < layerCount; ++i)
            {
                char layerType[256] = {0};
                char layerName[256] = {0};
                int bottomCount = 0;
                int topCount = 0;
                SCAN_VALUE("%255s", layerType)
                SCAN_VALUE("%255s", layerName)
                SCAN_VALUE("%d", bottomCount)
                SCAN_VALUE("%d", topCount)
                Layer::ptr layer = CreateLayer(layerType);
                if (!layer)
                {
                    SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "layer " << layerType << " not exists or registered!";
                }
                layer->m_type = std::move(layerType);
                layer->m_name = std::move(layerName);
                layer->m_bottoms.resize(bottomCount);
                for (int j = 0; j < bottomCount; ++j)
                {
                    char bottomName[256];
                    SCAN_VALUE("%255s", bottomName)
                    int bottomBlobIndex = findBlobIndexByName(bottomName);
                    if (bottomBlobIndex == -1)
                    {
                        Blob::ptr &blob = m_blobs[bottomBlobIndex];
                        bottomBlobIndex = blobIndex;
                        blob.reset(new Blob);
                        blob->m_name = std::move(bottomName);
                        blobIndex++;
                    }
                    Blob::ptr blob = m_blobs[bottomBlobIndex];
                    blob->m_consumers.push_back(i);
                    layer->m_bottoms[j] = bottomBlobIndex;
                }
                layer->m_tops.resize(topCount);
                for (int j = 0; j < topCount; ++j)
                {
                    char topName[256];
                    SCAN_VALUE("%255s", topName)
                    Blob::ptr &blob = m_blobs[blobIndex];
                    blob.reset(new Blob);
                    blob->m_producer = i;
                    blob->m_name = std::move(topName);
                    layer->m_tops[j] = blobIndex;
                    blobIndex++;
                }
                // layer specific params
                pd->loadParam(dr);
                layer->loadParam(pd);
                m_layers[i] = layer;
            }
        }

        void loadModel(FILE *fp)
        {
            DataReaderFromStdio::ptr dr(new DataReaderFromStdio(fp));
            return loadModel(dr);
        }

        void loadModel(const DataReader::ptr &dr)
        {
            if (m_layers.empty())
            {
                SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "network graph not ready";
            }
            size_t layerCount = m_layers.size();
            ModelBinFromDataReader::ptr mb(new ModelBinFromDataReader(dr));

            for (size_t i = 0; i < layerCount; i++)
            {
                Layer::ptr layer = m_layers[i];

                //Here we found inconsistent content in the parameter file.
                if (!layer)
                {
                    SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "load_model error at layer "
                                                     << "parameter file has inconsistent content.";
                }

                layer->loadModel(mb);
            }
        }

        void loadParamBin(FILE *fp)
        {
            // TODO
        }

        void loadParamBin(const DataReader &dr)
        {
            // TODO
        }

        // unload network structure and weight data
        void clear()
        {
            m_blobs.clear();
            m_layers.clear();
            m_threads.clear();
        }

        int findBlobIndexByName(const char *name) const
        {
            for (size_t i = 0; i < m_blobs.size(); ++i)
            {
                if (m_blobs[i]->m_name == name)
                {
                    return static_cast<int>(i);
                }
            }
            SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "find_blob_index_by_name " << name << " failed";
            return -1;
        }

#if USE_CHECK
        void check(std::vector<Mission> &tops, Layer::ptr layer)
        {
            Mission &m = tops[0];
            std::string &layerName = layer->m_name;
            for (size_t i = 0; i < m.h_data.size(); ++i)
            {
                Tensor::ptr t = m.h_data[i];
                t->cpu_data();
                layer->streamsync();
                if (t->elemsize() == sizeof(ComplexType))
                {
                    CHECK_DATA((const ComplexType *)t->cpu_data(), t->width(), layerName);
                }
                else if (t->elemsize() == sizeof(unsigned char))
                {
                    CHECK_DATA((const unsigned char *)t->cpu_data(), t->width(), layerName);
                }
                else if (t->elemsize() == sizeof(double))
                {
                    CHECK_DATA((const double *)t->cpu_data(), t->width(), layerName);
                }
                else
                {
                    SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "t->elemsize() is not corresponding type conversion!";
                }
            }
        }
#endif

        void forward(Layer::ptr layer, const Option &opt)
        {
            std::vector<Mission> bottoms;
            std::vector<Mission> tops;
            size_t bottomSize = layer->m_bottoms.size();
            size_t topSize = layer->m_tops.size();
            profiler::ptr prof = layer->getProfiler();
            if (m_profile)
            {
                prof->turn_on();
                std::string str = "pipeline::forward::" + layer->m_type;
                prof->scope_start(str.c_str());
            }
            while (!layer->isTerminate())
            {
                if (m_profile)
                {
                    prof->scope_start("start");
                }
                tops.resize(topSize);
                if (bottomSize > 0)
                {
                    bottoms = layer->getMission();
                    if (layer->isTerminate())
                    {
                        if (m_profile)
                        {
                            prof->scope_end();
                        }
                        continue;
                    }
                    for (size_t i = 0; i < topSize; ++i)
                    {
                        tops[i].mid = bottoms[0].mid;
                        tops[i].intertype = bottoms[0].intertype;
                        tops[i].head_data = bottoms[0].head_data;
                    }
                    std::sort(bottoms.begin(), bottoms.end(), [](Mission &a, Mission &b)
                              { return a.blobIndex < b.blobIndex; });
                }

                layer->forward(bottoms, tops, opt);
                layer->streamsync();
                // SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "layer name = " << layer->m_name << " mission count = " << layer->mcount() << " mid = " << tops[0].mid;
#if USE_CHECK
                check(tops, layer);
#endif
                if (layer == m_layers.front() && tops[0].datatype == DataType::report)
                {
                    m_ex->pushMission(tops[0]);
                }
                else if (layer == m_layers.back())
                {
                    for (size_t i = 0; i < topSize; ++i)
                    {
                        m_ex->pushMission(tops[i]);
                    }
                }
                else
                {
                    for (size_t i = 0; i < topSize; ++i)
                    {
                        tops[i].blobIndex = layer->m_tops[i];
                        std::vector<int> consumers = m_blobs[layer->m_tops[i]]->m_consumers;
                        for (size_t j = 0; j < consumers.size(); ++j)
                        {
                            m_layers[consumers[j]]->pushMission(tops[i]);
                        }
                    }
                }

                bottoms.clear();
                tops.clear();
                if (m_profile)
                {
                    prof->scope_end();
                }
            }
            if (m_profile)
            {
                prof->scope_end();
                prof->turn_off();
                std::string str = "profiler_" + layer->m_type;
                prof->DumpProfile(str.c_str());
            }
        }

    public:
        std::vector<Layer::ptr> m_layers;
        std::vector<Blob::ptr> m_blobs;
        std::vector<std::shared_ptr<std::thread>> m_threads;
        Extractor::ptr m_ex;

    private:
        DISABLE_COPY_AND_ASSIGN(impl)
        Command::ptr m_cmd;
        bool m_profile;
    };

    pipeline::pipeline() : m_impl(new impl)
    {
        spnnruntime::LoggerMgr::GetInstance()->init();
    }

    pipeline::~pipeline()
    {
        m_impl->wait();
        delete m_impl;
    }

    void pipeline::loadParam(std::string &&param)
    {
        m_impl->loadParam(std::move(param));
    }

    void pipeline::loadModel(std::string &&model)
    {
        m_impl->loadModel(std::move(model));
    }

    void pipeline::load_param_bin(std::string &&protopath)
    {
        m_impl->load_param_bin(std::move(protopath));
    }

    void pipeline::start()
    {
        m_impl->start();
        SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "pipeline start!";
    }

    // stop layer task
    void pipeline::stop()
    {
        m_impl->stop();
        SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "pipeline stop!";
    }

    bool pipeline::sendLaunchCmd(const unsigned char *_ptr)
    {
        return m_impl->sendLaunchCmd(_ptr);
    }

    bool pipeline::sendControlCmd(const unsigned char *_ptr)
    {
        return m_impl->sendControlCmd(_ptr);
    }

    Extractor::ptr pipeline::getExtractor() const
    {
        return m_impl->m_ex;
    }

    void pipeline::enableProfiler()
    {
        m_impl->enableProfiler();
    }

    void pipeline::disableProfiler()
    {
        m_impl->disableProfiler();
    }
} // spnnruntime