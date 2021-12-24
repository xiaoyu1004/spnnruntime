#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include "SPNNDefine.h"
#include "utils/common.h"
#include "gpu.h"

#include <cstdlib>

#if defined(_MSC_VER)
// #define WIN32_LEAN_AND_MEAN
// #include <windows.h>
#endif

namespace spnnruntime
{
#define SPNN_MALLOC_ALIGN 16

#define SPNN_MALLOC_OVERREAD 64

    // Aligns a pointer to the specified number of bytes
    // ptr Aligned pointer
    // n Alignment size that must be a power of two
    template <typename _Tp>
    static inline _Tp *alignPtr(_Tp *ptr, int n = (int)sizeof(_Tp))
    {
        return (_Tp *)(((size_t)ptr + n - 1) & -n);
    }

    // Aligns a buffer size to the specified number of bytes
    // The function returns the minimum number that is greater or equal to sz and is divisible by n
    // sz Buffer size to align
    // n Alignment size that must be a power of two
    static inline size_t alignSize(size_t sz, int n)
    {
        return (sz + n - 1) & -n;
    }

    static inline void *cpuMalloc(size_t size, bool pinned = false)
    {
        if (pinned)
        {
            void *ptr = 0;
            CUDA_CHECK(cudaMallocHost(&ptr, size));
            return ptr;
        }
#if _MSC_VER
        return _aligned_malloc(size, SPNN_MALLOC_ALIGN);
#elif defined(__unix__) && _POSIX_C_SOURCE >= 200112L
        void *ptr = 0;
        if (posix_memalign(&ptr, SPNN_MALLOC_ALIGN, size + SPNN_MALLOC_OVERREAD))
            ptr = 0;
        return ptr;
#else
        unsigned char *udata = (unsigned char *)malloc(size + sizeof(void *) + SPNN_MALLOC_ALIGN + SPNN_MALLOC_OVERREAD);
        if (!udata)
            return 0;
        unsigned char **adata = alignPtr((unsigned char **)udata + 1, SPNN_MALLOC_ALIGN);
        adata[-1] = udata;
        return adata;
#endif
    }

    static inline void cpuFree(void *ptr, bool pinned = false)
    {
        if (ptr)
        {
            if (pinned)
            {
                CUDA_CHECK(cudaFreeHost(ptr));
                return;
            }
#if _MSC_VER
            _aligned_free(ptr);
#elif defined(__unix__) && _POSIX_C_SOURCE >= 200112L
            free(ptr);
#else
            unsigned char *udata = ((unsigned char **)ptr)[-1];
            free(udata);
#endif
        }
    }

    static inline void *gpuMalloc(size_t size, cudaStream_t *stream = 0)
    {
        unsigned char *ptr = nullptr;
        if (stream)
        {
            CUDA_CHECK(cudaMallocAsync((void **)&ptr, size, *stream));
        }
        else
        {
            CUDA_CHECK(cudaMalloc((void **)&ptr, size));
        }
        return ptr;
    }

    static inline void gpuFree(void *ptr, cudaStream_t *stream = 0)
    {
        if (ptr)
        {
            if (stream)
            {
                CUDA_CHECK(cudaFreeAsync(ptr, *stream));
            }
            else
            {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
    }

    enum class Device : std::int8_t
    {
        UNKNOW = -2,
        CPU = -1,
        GPU = 0
    };

    class Allocator
    {
    public:
        using ptr = std::shared_ptr<Allocator>;
        Allocator(Device d);
        virtual ~Allocator(){};

    public:
        virtual void *fastMalloc(size_t size, bool pinned = true) = 0;
        virtual void fastFree(void *ptr, bool pinned = true) = 0;

        Device device() const { return m_device; }
        void stream(cudaStream_t *stream) { m_stream = stream; }
        cudaStream_t *stream() { return m_stream; }

    protected:
        Device m_device;
        cudaStream_t *m_stream;

    private:
        DISABLE_COPY_AND_ASSIGN(Allocator)
    };

    class PoolAllocator : public Allocator
    {
    public:
        class impl;
        typedef std::shared_ptr<PoolAllocator> ptr;
        PoolAllocator(Device deviceType);
        ~PoolAllocator();

        // ratio range 0 ~ 1
        // default cr = 0.75
        void setSizeCompareRatio(float scr);

        // release all budgets immediately
        void clear();

        virtual void *fastMalloc(size_t size, bool pinned = true);
        virtual void fastFree(void *ptr, bool pinned = true);

    private:
        DISABLE_COPY_AND_ASSIGN(PoolAllocator)
        impl *const m_impl;
    };

    class UnlockedPoolAllocator : public Allocator
    {
    public:
        class impl;
        UnlockedPoolAllocator(Device deviceType);
        ~UnlockedPoolAllocator();

        // ratio range 0 ~ 1
        // default cr = 0.75
        void setSizeCompareRatio(float scr);

        // release all budgets immediately
        void clear();

        virtual void *fastMalloc(size_t size, bool pinned = true);
        virtual void fastFree(void *ptr, bool pinned = true);

    private:
        DISABLE_COPY_AND_ASSIGN(UnlockedPoolAllocator)
        impl *const m_impl;
    };
} // spnnruntime

#endif