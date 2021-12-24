#include "syncedmem.h"

#include <cstring>

namespace spnnruntime
{
    syncedmem::syncedmem() : m_cpu_ptr(nullptr), m_gpu_ptr(nullptr), m_cpu_allocator(nullptr), m_gpu_allocator(nullptr), m_size(0),
                             m_head(UNINITIALIZED), m_own_cpu_data(false), m_own_gpu_data(false)
    {
    }

    syncedmem::syncedmem(size_t size, Allocator *cpuAllocator, Allocator *gpuAllocator) : m_cpu_ptr(nullptr), m_gpu_ptr(nullptr), m_cpu_allocator(cpuAllocator), m_gpu_allocator(gpuAllocator),
                                                                                          m_size(size), m_head(UNINITIALIZED), m_own_cpu_data(false), m_own_gpu_data(false)
    {
    }

    syncedmem::~syncedmem()
    {
        if (m_cpu_ptr && m_own_cpu_data)
        {
            if (m_cpu_allocator)
            {
                m_cpu_allocator->fastFree(m_cpu_ptr);
            }
            else
            {
                cpuFree(m_cpu_ptr);
            }
        }

        if (m_gpu_ptr && m_own_gpu_data)
        {
            if (m_gpu_allocator)
            {
                m_gpu_allocator->fastFree(m_gpu_ptr);
            }
            else
            {
                gpuFree(m_gpu_ptr);
            }
        }
    }

    void syncedmem::cpuAllocator(Allocator *allocator)
    {
        if (!m_cpu_allocator)
        {
            m_cpu_allocator = allocator;
        }
    }

    void syncedmem::gpuAllocator(Allocator *allocator)
    {
        if (!m_gpu_allocator)
        {
            m_gpu_allocator = allocator;
        }
    }

    const void *syncedmem::cpu_data()
    {
        to_cpu();
        return m_cpu_ptr;
    }

    const void *syncedmem::gpu_data()
    {
        to_gpu();
        return m_gpu_ptr;
    }

    void *syncedmem::mutable_cpu_data()
    {
        to_cpu();
        m_head = HEAD_AT_CPU;
        return m_cpu_ptr;
    }

    void *syncedmem::mutable_gpu_data()
    {
        to_gpu();
        m_head = HEAD_AT_GPU;
        return m_gpu_ptr;
    }

    void syncedmem::to_cpu()
    {
        switch (m_head)
        {
        case UNINITIALIZED:
            if (m_cpu_allocator)
            {
                m_cpu_ptr = static_cast<void *>(m_cpu_allocator->fastMalloc(m_size));
            }
            else
            {
                m_cpu_ptr = cpuMalloc(m_size);
            }
            memset(m_cpu_ptr, 0, m_size);
            m_head = HEAD_AT_CPU;
            m_own_cpu_data = true;
            break;
        case HEAD_AT_GPU:
            if (m_cpu_ptr == nullptr)
            {
                if (m_cpu_allocator)
                {
                    m_cpu_ptr = static_cast<void *>(m_cpu_allocator->fastMalloc(m_size));
                }
                else
                {
                    m_cpu_ptr = cpuMalloc(m_size);
                }
                m_own_cpu_data = true;
            }
            if (m_gpu_ptr != m_cpu_ptr)
            {
                if (m_gpu_allocator)
                {
                    CUDA_CHECK(cudaMemcpyAsync(m_cpu_ptr, m_gpu_ptr, m_size, cudaMemcpyDeviceToHost, *m_gpu_allocator->stream()));
                }
                else
                {
                    CUDA_CHECK(cudaMemcpy(m_cpu_ptr, m_gpu_ptr, m_size, cudaMemcpyDeviceToHost));
                }
            }
            m_head = SYNCED;
            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
        }
    }

    void syncedmem::to_gpu()
    {
        switch (m_head)
        {
        case UNINITIALIZED:
            if (m_gpu_allocator)
            {
                m_gpu_ptr = static_cast<void *>(m_gpu_allocator->fastMalloc(m_size));
                CUDA_CHECK(cudaMemsetAsync(m_gpu_ptr, 0, m_size, *m_gpu_allocator->stream()));
            }
            else
            {
                m_gpu_ptr = gpuMalloc(m_size);
                CUDA_CHECK(cudaMemset(m_gpu_ptr, 0, m_size));
            }
            m_head = HEAD_AT_GPU;
            m_own_gpu_data = true;
            break;
        case HEAD_AT_CPU:
            if (m_gpu_ptr == nullptr)
            {
                if (m_gpu_allocator)
                {
                    m_gpu_ptr = static_cast<void *>(m_gpu_allocator->fastMalloc(m_size));
                    CUDA_CHECK(cudaMemcpyAsync(m_gpu_ptr, m_cpu_ptr, m_size, cudaMemcpyHostToDevice, *m_gpu_allocator->stream()));
                }
                else
                {
                    m_gpu_ptr = gpuMalloc(m_size);
                    CUDA_CHECK(cudaMemcpy(m_gpu_ptr, m_cpu_ptr, m_size, cudaMemcpyHostToDevice));
                }
                m_own_gpu_data = true;
            }
            m_head = SYNCED;
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
        }
    }
} // spnnruntime