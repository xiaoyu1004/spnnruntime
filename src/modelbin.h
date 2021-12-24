#ifndef MODELBIN_H
#define MODELBIN_H

#include "SPNNDefine.h"
#include "utils/common.h"
#include "tensor.h"
#include "datareader.h"

namespace spnnruntime
{
    class SPNN_EXPORT ModelBin
    {
    public:
        typedef std::shared_ptr<ModelBin> ptr;
        ModelBin() {}
        virtual ~ModelBin() {}

        // element type
        // 1 = float32
        // 2 = std::complex<double>
        // load vec
        virtual Tensor::ptr load(int w, int type) const = 0;
        // load 2_dim
        Tensor::ptr load(int w, int h, int type) const;
    };

    class ModelBinFromDataReader : public ModelBin
    {
    public:
        typedef std::shared_ptr<ModelBinFromDataReader> ptr;
        ModelBinFromDataReader(const DataReader::ptr dr);
        virtual ~ModelBinFromDataReader();

        virtual Tensor::ptr load(int w, int type) const;

    private:
        DISABLE_COPY_AND_ASSIGN(ModelBinFromDataReader)
        const DataReader ::ptr m_dr;
    };
}

#endif