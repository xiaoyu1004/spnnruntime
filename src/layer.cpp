#include "layer.h"
#include "log.h"

#include <cstring>

namespace spnnruntime
{
#include "layer_declaration.h"
#include "layer_registry.h"

    static const int LayerRegistryEntryCount = sizeof(LayerRegistry) / sizeof(LayerRegistryEntry);

    // load layer specific parameter from parsed dict
    void Layer::loadParam(const ParamDict::ptr pd)
    {
        // do nothing
    }

    // load layer specific weight data from model binary
    void Layer::loadModel(const ModelBin::ptr mb)
    {
        // do nothing
    }

    std::vector<Mission> Layer::getMission()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        size_t count = m_bottoms.size();
        std::vector<Mission> missions;
        if (count == 1)
        {
            m_condition.wait(lock, [this]()
                             { return !m_missions.empty() || m_terminate; });
            if (m_terminate)
            {
                return std::vector<Mission>{};
            }
            Mission m = m_missions.front();
            m_missions.pop_front();
            missions.push_back(m);
        }
        else if (count >= 2)
        {
            m_condition.wait(lock, [&]()
                             {
                                 if (m_terminate)
                                 {
                                     return true;
                                 }
                                 size_t cnt = 0;
                                 if (m_missions.empty() || m_terminate)
                                 {
                                     return false;
                                 }
                                 std::vector<std::list<Mission>::iterator> vec;
                                 Mission m = m_missions.front();
                                 std::list<Mission>::iterator begin = m_missions.begin();
                                 while (begin != m_missions.end())
                                 {
                                     if (begin->mid == m.mid)
                                     {
                                         missions.push_back(*begin);
                                         vec.push_back(begin);
                                         cnt++;
                                     }
                                     if (cnt == count)
                                     {
                                         for (size_t i = 0; i < vec.size(); ++i)
                                         {
                                             m_missions.erase(vec[i]);
                                         }
                                         return true;
                                     }
                                     begin++;
                                 }
                                 missions.clear();
                                 return false;
                             });
            if (m_terminate)
            {
                return std::vector<Mission>{};
            }
        }
        return missions;
    }

    void Layer::pushMission(Mission &m)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_missions.push_back(m);
        lock.unlock();
        m_condition.notify_one();
    }

    // create layer from type name
    Layer::ptr CreateLayer(const char *type)
    {
        int index = LayerToIndex(type);
        if (index == -1)
            return nullptr;
        return CreateLayer(index);
    }

    // get layer type from type name
    int LayerToIndex(const char *type)
    {
        for (int i = 0; i < LayerRegistryEntryCount; i++)
        {
            if (strcmp(type, LayerRegistry[i].name) == 0)
            {
                return i;
            }
        }
        SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "[" << type << "] is not found in layer registry list!";
        return -1;
    }

    // create layer from layer type
    Layer::ptr CreateLayer(int index)
    {
        if (index < 0 || index >= LayerRegistryEntryCount)
        {
            SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "[index=" << index << "] is invalid!";
        }
        LayerCreatorFunc layerCreator = LayerRegistry[index].creator;
        if (!layerCreator)
        {
            SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "error! layerCreator cannot be nullptr!";
        }
        Layer::ptr layer(layerCreator(0));
        layer->m_typeindex = index;
        return layer;
    }
} // spnnruntime