#ifndef BLOB_H
#define BLOB_H

#include "SPNNDefine.h"
#include "utils/common.h"
#include "tensor.h"

#include <string>
#include <vector>

namespace spnnruntime
{
    class Blob
    {
    public:
        typedef std::shared_ptr<Blob> ptr;
        // empty
        Blob()
        {
            m_producer = -1;
        }

    public:
        // blob name
        std::string m_name;
        // layer index which produce this blob as output
        int m_producer;
        // layer index which need this blob as input
        std::vector<int> m_consumers;
    };
}

#endif