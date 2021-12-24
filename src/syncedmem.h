#ifndef SYNCEDMEM_H
#define SYNCEDMEM_H

#include "allocator.h"
#include "gpu.h"

namespace spnnruntime
{
    class syncedmem
    {
    public:
        typedef std::shared_ptr<syncedmem> ptr;
        syncedmem();
        explicit syncedmem(size_t size, Allocator * cpuAllocator = 0, Allocator * gpuAllocator = 0);
        ~syncedmem();

        enum SyncedHead
        {
            UNINITIALIZED,
            HEAD_AT_CPU,
            HEAD_AT_GPU,
            SYNCED
        };
        SyncedHead head() { return m_head; }
        bool empty() const { return m_size == 0; }

        void cpuAllocator(Allocator *allocator);
        void gpuAllocator(Allocator *allocator);

        size_t size() { return m_size; }
        const void *cpu_data();
        const void *gpu_data();
        void *mutable_cpu_data();
        void *mutable_gpu_data();

    private:
        void *m_cpu_ptr;
        void *m_gpu_ptr;
        Allocator *m_cpu_allocator;
        Allocator *m_gpu_allocator;
        size_t m_size;
        SyncedHead m_head;
        bool m_own_cpu_data;
        bool m_own_gpu_data;

        void to_cpu();
        void to_gpu();
    };
} // spnnruntime

#endif