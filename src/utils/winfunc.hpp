#ifndef WINFUNC_H
#define WINFUNC_H

#include "tensor.h"

#include <vector>
#define PI 3.1415926

namespace spnnruntime
{
    /**
 * Calculates hannWin window coefficients.
 */
    template <typename Type>
    Tensor::ptr hanning(std::uint32_t N, Type amp)
    {
        Tensor::ptr win(new Tensor(N, sizeof(Type), nullptr, nullptr));

        for (std::uint32_t i = 0; i < (N + 1) / 2; ++i)
        {
            ((Type *)win->mutable_cpu_data())[i] = amp * Type(0.5 - 0.5 * cos(2 * PI * i / (N - 1)));
            ((Type *)win->mutable_cpu_data())[N - 1 - i] = ((Type *)win->cpu_data())[i];
        }

        return win;
    }

    /**
 * Calculates hamming window coefficients.
 */
    template <typename Type>
    Tensor::ptr hamming(std::uint32_t N, Type amp)
    {
        Tensor::ptr win(new Tensor(N, sizeof(Type), nullptr, nullptr));

        for (std::uint32_t i = 0; i < (N + 1) / 2; ++i)
        {
            ((Type *)win->mutable_cpu_data())[i] = amp * Type(0.54 - 0.46 * cos(2 * PI * i / (N - 1.0)));
            ((Type *)win->mutable_cpu_data())[N - 1 - i] = ((Type *)win->cpu_data())[i];
        }

        return win;
    }
} // spnnruntime

#endif