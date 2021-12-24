#ifndef PARAMDICT_H
#define PARAMDICT_H

#include "SPNNDefine.h"
#include "utils/common.h"
#include "tensor.h"
#include "datareader.h"

#include <memory>
#include <vector>

// at most 32 parameters
#define SPNN_MAX_PARAM_COUNT 32

namespace spnnruntime
{
    class ParamDict
    {
    public:
        typedef std::shared_ptr<ParamDict> ptr;
        // empty
        ParamDict();

    public:
        virtual ~ParamDict() {};

        // get type
        int type(int id) const;

        // get int
        int get(int id, int def) const;
        // get float
        float get(int id, float def) const;
        // get array
        Tensor::ptr get(int id, const Tensor::ptr def) const;

        // set int
        void set(int id, int i);
        // set float
        void set(int id, float f);
        // set array
        void set(int id, const Tensor::ptr v);

        void clear();

        void loadParam(const DataReader::ptr dr);
        void loadParamBin(const DataReader::ptr dr);

    private:
        struct Param
        {
            // 0 = null
            // 1 = int/float
            // 2 = int
            // 3 = float
            // 4 = array of int/float
            // 5 = array of int
            // 6 = array of float
            int type;
            union
            {
                int i;
                float f;
            };
            Tensor::ptr v;
        };
        std::vector<Param> params;
    };
} // spnnruntime

#endif