#ifndef CHECK_HPP
#define CHECK_HPP

#include "allocator.h"
#include "setting/params.h"
#include "gpu.h"
#include "log.h"

#include <algorithm>

namespace spnnruntime
{
#if USE_CHECK

#define CHECK_READ(ex)                                                         \
    if (ex == (std::uint32_t)-1)                                               \
    {                                                                          \
        SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << __func__ << " : read file error!"; \
    }

#define CHECK_DATA(data, len, file)                                                  \
    if (std::count(g_pulse_lens.begin(), g_pulse_lens.end(), len))                   \
    {                                                                                \
        CheckData(data, len, file);                                                  \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "layer: " << file << " skip data check!"; \
    }

    inline std::ostream &operator<<(std::ostream &os, const ComplexType &b)
    {
        os << "(" << b.x << "|" << b.y << ")";
        return os;
    }

    inline bool operator==(const ComplexType &a, const ComplexType &b)
    {
        return cuCabs(cuCsub(a, b)) < 1e-5f;
    }

    inline bool operator!=(const ComplexType &a, const ComplexType &b)
    {
        return cuCabs(cuCsub(a, b)) >= 1e-5f;
    }

    inline double operator-(const ComplexType &a, const ComplexType &b)
    {
        return std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2);
    }

    inline double operator*(const ComplexType &a, const ComplexType &b)
    {
        return std::pow(a.x, 2) + std::pow(b.y, 2);
    }

    template <typename T>
    static inline bool CompareVal(const T *a, const T *b, const std::uint32_t len, const std::string &file, const double epsilon)
    {
        for (std::uint32_t i = 0; i < len; ++i)
        {
            if ((a[i] - b[i]) >= 1e-2f)
            {
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ERROR! a[" << i << "]" << a[i] << " != b[" << i << "]" << b[i] << "  error is : " << a[i] - b[i] << "  layer : " << file;
                return false;
            }
        }
        return true;
    }

    template <typename T>
    inline bool CompareL2fe(const T *reference, const T *data, const std::uint32_t len, const double epsilon)
    {
        assert(epsilon >= 0);

        double error = 0;
        double ref = 0;

        for (std::uint32_t i = 0; i < len; ++i)
        {
            double diff = reference[i] - data[i];
            error += diff;
            ref += reference[i] * reference[i];
        }

        double normRef = sqrtf(ref);

        if (fabs(ref) < 1e-7)
        {
            SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "WARN, reference l2-norm is 0";
            return true;
        }

        double normError = sqrtf(error);
        error = normError / normRef;
        bool result = error < epsilon;
        if (!result)
        {
            SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ERROR, l2-norm error " << error << " is greater than epsilon " << epsilon;
        }
        return result;
    }

    template <typename T>
    static void CheckData(const T *data, const std::uint32_t len, const std::string &file)
    {
        std::string filepath = "/home/yunjian/projects/spnnruntime/dataset/" + file + "_" + std::to_string(len) + "_data.dat";
        FILE *fd = fopen(filepath.c_str(), "rb");
        std::uint32_t length, number;
        CHECK_READ(fread(&length, sizeof(std::uint32_t), 1, fd));
        CHECK_READ(fread(&number, sizeof(std::uint32_t), 1, fd));
        const std::uint32_t N = length * number;
        T *f_data = new T[N];
        CHECK_READ(fread(f_data, sizeof(T), N, fd));
        fclose(fd);
        // bool bTestResult = CompareL2fe<T>(f_data, data, N, 1e-5f);
        bool bTestResult = CompareVal<T>(f_data, data, N, file, 1e-5f);
        delete[] f_data;
        if (bTestResult)
        {
            SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "check [" << filepath << "] data pass!";
            return;
        }
        SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "check [" << filepath << "] data failed!";
    }
#else
#define CHECK_DATA(data, length, file)
#endif
} // spnnruntime

#endif