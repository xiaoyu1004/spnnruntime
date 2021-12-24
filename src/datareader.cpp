#include "datareader.h"

#include <cstdio>
#include <cstdarg>

namespace spnnruntime
{
    DataReaderFromStdio::DataReaderFromStdio(FILE *fp) : m_fp(fp)
    {
    }

    bool DataReaderFromStdio::scan(const char *format, void *p) const
    {
        return fscanf(m_fp, format, p) == 1;
    }

    size_t DataReaderFromStdio::read(void *buf, size_t size) const
    {
        return fread(buf, 1, size, m_fp);
    }
} // spnnruntime