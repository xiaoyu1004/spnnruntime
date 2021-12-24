#ifndef DATAREADER_H
#define DATAREADER_H

#include "SPNNDefine.h"
#include "utils/common.h"

#include <memory>

namespace spnnruntime
{
    class DataReader
    {
    public:
        typedef std::shared_ptr<DataReader> ptr;
        DataReader() {};
        virtual ~DataReader() {};

    public:
        // parse plain param text
        // return true if scan success
        virtual bool scan(const char *format, void *p) const = 0;

        // read binary param and model data
        // return bytes read
        virtual size_t read(void *buf, size_t size) const = 0;

        // get model data reference
        // return bytes referenced
        // virtual size_t reference(size_t size, const void **buf) const = 0;

    private:
        DISABLE_COPY_AND_ASSIGN(DataReader)
    };

    class DataReaderFromStdio : public DataReader
    {
    public:
        typedef std::shared_ptr<DataReaderFromStdio> ptr;
        explicit DataReaderFromStdio(FILE *fp);
        virtual ~DataReaderFromStdio() {};

        virtual bool scan(const char *format, void *p) const override;
        virtual size_t read(void *buf, size_t size) const override;

    private:
        DISABLE_COPY_AND_ASSIGN(DataReaderFromStdio)
        FILE *m_fp;
    };
}

#endif