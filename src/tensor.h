#ifndef TENSOR_H
#define TENSOR_H

#include "SPNNDefine.h"
#include "allocator.h"
#include "syncedmem.h"

#include <cstdint>
#include <memory>

namespace spnnruntime
{
    class SPNN_EXPORT Tensor final
    {
    public:
        typedef std::shared_ptr<Tensor> ptr;
        Tensor();
        Tensor(const Tensor &);
        Tensor(std::uint32_t w, std::uint32_t elemsize = 4u, Allocator * cpuAllocator = 0, Allocator * gpuAllocator = 0);
        Tensor(std::uint32_t w, std::uint32_t h, std::uint32_t elemsize = 4u, Allocator * cpuAllocator = 0, Allocator * gpuAllocator = 0);
        ~Tensor();

    public:
        Tensor &operator=(const Tensor &);
        void fill(double v);
        void fill(float v);
        void fill(int v);
        template <typename T>
        void fill(T v);

        bool empty() const;
        std::uint32_t count() const;
        // shape only
        const std::vector<std::uint32_t> shape() const;

        void reshape(std::uint32_t w);
        void reshape(std::uint32_t w, std::uint32_t h);

        float *row(int y);
        const float *row(int y) const;

        template <typename T>
        T *row(int y);
        
        template <typename T>
        const T *row(int y) const;

        // access raw data
        template <typename T>
        operator T *()
        {
            return (T *)m_data->mutable_cpu_data();
        }

        template <typename T>
        operator const T *() const
        {
            return (const T *)m_data->cpu_data();
        }

        // convenient access float vec element
        float &operator[](size_t i);
        const float &operator[](size_t i) const;

        const void *cpu_data() const;
        const void *gpu_data() const;
        void *mutable_cpu_data();
        void *mutable_gpu_data();

        std::uint32_t dims() const { return m_dims; }
        std::uint32_t width() const { return m_w; }
        std::uint32_t height() const { return m_h; }
        size_t elemsize() const { return m_elemsize; }

    private:
        syncedmem *m_data;

        // element size in bytes
        // 8 = double
        // 4 = float32/int32
        // 2 = float16
        // 1 = int8/uint8
        // 0 = empty
        std::uint32_t m_elemsize;

        // the allocator
        Allocator * m_cpu_allocator;
        Allocator * m_gpu_allocator;

        // the dimension rank
        std::uint32_t m_dims;

        std::uint32_t m_w;
        std::uint32_t m_h;
    };
} // spnnruntime

#endif