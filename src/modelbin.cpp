#include "modelbin.h"
#include "datareader.h"

#include <complex>

namespace spnnruntime
{
    Tensor::ptr ModelBin::load(int w, int h, int type) const
    {
        Tensor::ptr t = load(w * h, type);
        if (t->empty())
        {
            return t;
        }

        t->reshape(w, h);
        return t;
    }

    ModelBinFromDataReader::ModelBinFromDataReader(const DataReader::ptr dr) : m_dr(dr)
    {
    }

    ModelBinFromDataReader::~ModelBinFromDataReader()
    {
    }

    Tensor::ptr ModelBinFromDataReader::load(int w, int type) const
    {
        Tensor::ptr t;

        // raw data
        if (type == 1) // float32
        {
            t.reset(new Tensor(w));
            size_t nread = m_dr->read(t->mutable_cpu_data(), w * sizeof(float));
            if (nread != w * sizeof(float))
            {
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ModelBin read weight_data failed " << nread;
                t.reset(new Tensor);
                return t;
            }
        }
        else if (type == 2) // std::complex<double>
        {
            t.reset(new Tensor(w, sizeof(std::complex<double>), nullptr, nullptr));
            size_t nread = m_dr->read(t->mutable_cpu_data(), w * sizeof(std::complex<double>));
            if (nread != w * sizeof(std::complex<double>))
            {
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ModelBin read weight_data failed " << nread;
                t.reset(new Tensor);
                return t;
            }
        }
        else
        {
            SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ModelBin load type " << type << " not implemented";
            t.reset(new Tensor);
            return t;
        }
        return t;
    }
} // spnnruntime