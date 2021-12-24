#include "tensor.h"

namespace spnnruntime
{
    Tensor::Tensor() : m_data(0), m_elemsize(0), m_cpu_allocator(0), m_gpu_allocator(0), m_dims(0), m_w(0), m_h(0)
    {
    }

    Tensor::Tensor(std::uint32_t w, std::uint32_t elemsize, Allocator *cpuAllocator, Allocator *gpuAllocator) : m_data(0), m_elemsize(0), m_cpu_allocator(0), m_gpu_allocator(0), m_dims(0), m_w(0), m_h(0)
    {
        if (m_dims == 1 && m_w == w && m_elemsize == elemsize && m_cpu_allocator == cpuAllocator && m_gpu_allocator == gpuAllocator)
        {
            return;
        }

        m_elemsize = elemsize;
        m_cpu_allocator = cpuAllocator;
        m_gpu_allocator = gpuAllocator;

        m_dims = 1;
        m_w = w;
        m_h = 1;

        m_data = new syncedmem(w * elemsize, m_cpu_allocator, m_gpu_allocator);
    }

    Tensor::Tensor(std::uint32_t w, std::uint32_t h, std::uint32_t elemsize, Allocator *cpuAllocator, Allocator *gpuAllocator) : m_data(0), m_elemsize(0), m_cpu_allocator(0), m_gpu_allocator(0), m_dims(0), m_w(0), m_h(0)
    {
        if (m_dims == 2 && m_w == w && m_h == h && m_elemsize == elemsize && m_cpu_allocator == cpuAllocator && m_gpu_allocator == gpuAllocator)
        {
            return;
        }

        m_elemsize = elemsize;
        m_cpu_allocator = cpuAllocator;
        m_gpu_allocator = gpuAllocator;

        m_dims = 2;
        m_w = w;
        m_h = h;

        m_data = new syncedmem(w * h * elemsize, m_cpu_allocator, m_gpu_allocator);
    }

    Tensor::Tensor(const Tensor &t)
    {
        m_elemsize = t.m_elemsize;
        m_cpu_allocator = t.m_cpu_allocator;
        m_gpu_allocator = t.m_gpu_allocator;

        m_dims = t.m_dims;
        m_w = t.m_w;
        m_h = t.m_h;

        m_data = t.m_data;
        m_data->cpuAllocator(t.m_cpu_allocator);
        m_data->gpuAllocator(t.m_gpu_allocator);
    }

    Tensor &Tensor::operator=(const Tensor &t)
    {
        if (this == &t)
            return *this;

        m_data = t.m_data;
        m_elemsize = t.m_elemsize;
        m_cpu_allocator = t.m_cpu_allocator;
        m_gpu_allocator = t.m_gpu_allocator;

        m_dims = t.m_dims;
        m_w = t.m_w;
        m_h = t.m_h;

        return *this;
    }

    Tensor::~Tensor()
    {
        delete m_data;
    };

    void Tensor::fill(double v)
    {
        int size = (int)count();
        double *ptr = (double *)m_data->cpu_data();
        for (; size > 0; size--)
        {
            *ptr++ = v;
        }
    }

    void Tensor::fill(float v)
    {
        int size = (int)count();
        float *ptr = (float *)m_data->cpu_data();
        for (; size > 0; size--)
        {
            *ptr++ = v;
        }
    }

    void Tensor::fill(int v)
    {
        int size = (int)count();
        int *ptr = (int *)m_data->cpu_data();
        for (; size > 0; size--)
        {
            *ptr++ = v;
        }
    }

    template <typename T>
    void Tensor::fill(T v)
    {
        int size = (int)count();
        T *ptr = (T *)m_data->cpu_data();
        for (int i = 0; i < size; i++)
        {
            ptr[i] = v;
        }
    }

    bool Tensor::empty() const
    {
        return m_data->empty() || count() == 0;
    }

    std::uint32_t Tensor::count() const
    {
        return m_w * m_h;
    }

    // shape
    const std::vector<std::uint32_t> Tensor::shape() const
    {
        return std::vector<std::uint32_t>{m_w, m_h};
    }

    void Tensor::reshape(std::uint32_t w)
    {
        CHECK_EQ(w, m_w * m_h);
        m_w = w;
        m_h = 1;
    }

    void Tensor::reshape(std::uint32_t w, std::uint32_t h)
    {
        CHECK_EQ(w * h, m_w * m_h);
        m_w = w;
        m_h = h;
    }

    float *Tensor::row(int y)
    {
        return (float *)((unsigned char *)m_data->cpu_data() + m_w * y * m_elemsize);
    }

    const float *Tensor::row(int y) const
    {
        return (const float *)((unsigned char *)m_data->cpu_data() + m_w * y * m_elemsize);
    }

    template <typename T>
    T *Tensor::row(int y)
    {
        return (T *)((unsigned char *)m_data->cpu_data() + m_w * y * m_elemsize);
    }

    template <typename T>
    const T *Tensor::row(int y) const
    {
        return (const T *)((unsigned char *)m_data->cpu_data() + m_w * y * m_elemsize);
    }

    // convenient access float vec element
    float &Tensor::operator[](size_t i)
    {
        return ((float *)m_data->cpu_data())[i];
    }

    const float &Tensor::operator[](size_t i) const
    {
        return ((const float *)m_data->cpu_data())[i];
    }

    const void *Tensor::cpu_data() const
    {
        return m_data->cpu_data();
    }

    void *Tensor::mutable_cpu_data()
    {
        return m_data->mutable_cpu_data();
    }

    const void *Tensor::gpu_data() const
    {
        return m_data->gpu_data();
    }

    void *Tensor::mutable_gpu_data()
    {
        return m_data->mutable_gpu_data();
    }
} // spnnruntime