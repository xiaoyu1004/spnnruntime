#include "allocator.h"

#include <mutex>
#include <list>

namespace spnnruntime
{
    Allocator::Allocator(Device d) : m_device(d)
    {
    }

    class PoolAllocator::impl
    {
    public:
        impl() {}

    public:
        std::mutex m_budgets_lock;
        std::mutex m_payouts_lock;
        unsigned int m_size_compare_ratio; // 0~256
        std::list<std::pair<size_t, void *>> m_budgets;
        std::list<std::pair<size_t, void *>> m_payouts;

    private:
        DISABLE_COPY_AND_ASSIGN(impl)
    };

    PoolAllocator::PoolAllocator(Device d) : Allocator(d), m_impl(new impl)
    {
        m_impl->m_size_compare_ratio = 192; // 0.75f * 256
    }

    PoolAllocator::~PoolAllocator()
    {
        clear();

        if (!m_impl->m_payouts.empty())
        {
            auto it = m_impl->m_payouts.begin();
            for (; it != m_impl->m_payouts.end(); ++it)
            {
                void *ptr = it->second;
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << ptr << " still in use";
            }
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "FATAL ERROR! pool allocator destroyed too early";
        }
        delete m_impl;
    }

    // ratio range 0 ~ 1
    // default cr = 0.75
    void PoolAllocator::setSizeCompareRatio(float scr)
    {
        if (scr < 0.f || scr > 1.f)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "invalid size compare ratio " << scr;
            return;
        }

        m_impl->m_size_compare_ratio = (unsigned int)(scr * 256);
    }

    // release all budgets immediately
    void PoolAllocator::clear()
    {
        m_impl->m_budgets_lock.lock();

        auto it = m_impl->m_budgets.begin();
        for (; it != m_impl->m_budgets.end(); ++it)
        {
            void *ptr = it->second;
            if (m_device == Device::CPU)
            {
                spnnruntime::cpuFree(ptr, true);
            }
            else if (m_device == Device::GPU)
            {
                spnnruntime::gpuFree(ptr);
            }
        }
        m_impl->m_budgets.clear();

        m_impl->m_budgets_lock.unlock();
    }

    void *PoolAllocator::fastMalloc(size_t size, bool pinned)
    {
        m_impl->m_budgets_lock.lock();

        // find free budget
        auto it = m_impl->m_budgets.begin();
        for (; it != m_impl->m_budgets.end(); ++it)
        {
            size_t bs = it->first;

            // size_compare_ratio ~ 100%
            if (bs >= size && ((bs * m_impl->m_size_compare_ratio) >> 8) <= size)
            {
                void *ptr = it->second;

                m_impl->m_budgets.erase(it);

                m_impl->m_budgets_lock.unlock();

                m_impl->m_payouts_lock.lock();

                m_impl->m_payouts.push_back(std::make_pair(bs, ptr));

                m_impl->m_payouts_lock.unlock();

                return ptr;
            }
        }

        m_impl->m_budgets_lock.unlock();

        // new
        void *ptr = nullptr;
        if (m_device == Device::CPU)
        {
            ptr = spnnruntime::cpuMalloc(size, pinned);
        }
        else if (m_device == Device::GPU)
        {
            ptr = spnnruntime::gpuMalloc(size, m_stream);
        }

        m_impl->m_payouts_lock.lock();

        m_impl->m_payouts.push_back(std::make_pair(size, ptr));

        m_impl->m_payouts_lock.unlock();

        return ptr;
    }

    void PoolAllocator::fastFree(void *ptr, bool pinned)
    {
        m_impl->m_payouts_lock.lock();

        // return to budgets
        for (auto it = m_impl->m_payouts.begin(); it != m_impl->m_payouts.end(); ++it)
        {
            if (it->second == ptr)
            {
                size_t size = it->first;

                m_impl->m_payouts.erase(it);

                m_impl->m_payouts_lock.unlock();

                m_impl->m_budgets_lock.lock();

                m_impl->m_budgets.push_back(std::make_pair(size, ptr));

                m_impl->m_budgets_lock.unlock();

                return;
            }
        }

        m_impl->m_payouts_lock.unlock();

        SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "FATAL ERROR! pool allocator get wild " << ptr;

        if (m_device == Device::CPU)
        {
            spnnruntime::cpuFree(ptr, pinned);
        }
        else if (m_device == Device::GPU)
        {
            spnnruntime::gpuFree(ptr, m_stream);
        }
    }

    class UnlockedPoolAllocator::impl
    {
    public:
        impl() {}

    public:
        unsigned int m_size_compare_ratio; // 0~256
        std::list<std::pair<size_t, void *>> m_budgets;
        std::list<std::pair<size_t, void *>> m_payouts;

    private:
        DISABLE_COPY_AND_ASSIGN(impl)
    };

    UnlockedPoolAllocator::UnlockedPoolAllocator(Device deviceType)
        : Allocator(deviceType), m_impl(new impl)
    {
        m_impl->m_size_compare_ratio = 192; // 0.75f * 256
    }

    UnlockedPoolAllocator::~UnlockedPoolAllocator()
    {
        clear();

        if (!m_impl->m_payouts.empty())
        {
            auto it = m_impl->m_payouts.begin();
            for (; it != m_impl->m_payouts.end(); ++it)
            {
                void *ptr = it->second;
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << ptr << " still in use";
            }
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "FATAL ERROR! unlocked pool allocator destroyed too early";
        }

        delete m_impl;
    }

    void UnlockedPoolAllocator::clear()
    {
        auto it = m_impl->m_budgets.begin();
        for (; it != m_impl->m_budgets.end(); ++it)
        {
            void *ptr = it->second;
            if (m_device == Device::CPU)
            {
                spnnruntime::cpuFree(ptr, true);
            }
            else if (m_device == Device::GPU)
            {
                spnnruntime::gpuFree(ptr, m_stream);
            }
        }
        m_impl->m_budgets.clear();
    }

    void UnlockedPoolAllocator::setSizeCompareRatio(float scr)
    {
        if (scr < 0.f || scr > 1.f)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "invalid size compare ratio " << scr;
            return;
        }

        m_impl->m_size_compare_ratio = (unsigned int)(scr * 256);
    }

    void *UnlockedPoolAllocator::fastMalloc(size_t size, bool pinned)
    {
        // find free budget
        auto it = m_impl->m_budgets.begin();
        for (; it != m_impl->m_budgets.end(); ++it)
        {
            size_t bs = it->first;

            // size_compare_ratio ~ 100%
            if (bs >= size && ((bs * m_impl->m_size_compare_ratio) >> 8) <= size)
            {
                void *ptr = it->second;

                m_impl->m_budgets.erase(it);

                m_impl->m_payouts.push_back(std::make_pair(bs, ptr));

                return ptr;
            }
        }

        // new
        void *ptr = nullptr;
        if (m_device == Device::CPU)
        {
            ptr = spnnruntime::cpuMalloc(size, pinned);
        }
        else if (m_device == Device::GPU)
        {
            ptr = spnnruntime::gpuMalloc(size, m_stream);
        }

        m_impl->m_payouts.push_back(std::make_pair(size, ptr));

        return ptr;
    }

    void UnlockedPoolAllocator::fastFree(void *ptr, bool pinned)
    {
        // return to budgets
        for (auto it = m_impl->m_payouts.begin(); it != m_impl->m_payouts.end(); ++it)
        {
            if (it->second == ptr)
            {
                size_t size = it->first;

                m_impl->m_payouts.erase(it);

                m_impl->m_budgets.push_back(std::make_pair(size, ptr));

                return;
            }
        }

        SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "FATAL ERROR! pool allocator get wild " << ptr;

        if (m_device == Device::CPU)
        {
            spnnruntime::cpuFree(ptr, pinned);
        }
        else if (m_device == Device::GPU)
        {
            spnnruntime::gpuFree(ptr, m_stream);
        }
    }
} // spnnruntime