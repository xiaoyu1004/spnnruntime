#ifndef COMMAND_H
#define COMMAND_H

#include "log.h"
#include "setting/radar.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <memory>

#if __BYTE_ORDER == __LITTLE_ENDIAN
#define ltohl(x) (x)
#define ltohs(x) (x)
#define htoll(x) (x)
#define htols(x) (x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#define ltohl(x) __bswap_32(x)
#define ltohs(x) __bswap_16(x)
#define htoll(x) __bswap_32(x)
#define htols(x) __bswap_16(x)
#endif

#define MAP_SIZE (1024 * 1024UL)
#define MAP_MASK (MAP_SIZE - 1)

namespace spnnruntime
{
    class Command
    {
    public:
        friend class Input;
        typedef std::shared_ptr<Command> ptr;
        Command();

    public:
        void loadData(void *_ptr, size_t size);
        std::uint32_t getInterType();
        DataType getDataType(std::uint32_t intertype) const;
        bool sendLaunchCmd(const unsigned char *_ptr);
        bool sendControlCmd(const unsigned char *_ptr);

    private:
        int openEvent(const char *filename);
        int readEvent(int fd);
        int openControl(const char *filename);
        void *mmapControl(int fd, long mapsize);
        void writeControl(void *base_addr, int offset, std::uint32_t val);
        std::uint32_t readControl(void *base_addr, int offset);

        std::string getFilePath();

    private:
        DISABLE_COPY_AND_ASSIGN(Command)
        unsigned char *m_head;
        unsigned char *m_write_data;
        void *m_control_base;
        int m_control_fd;
        int m_fd_event;
        int m_fd_c2h;
        int m_fd_h2c;

        int m_fd;
    };

    typedef SingletonPtr<Command> CmdMgr;
} // spnnruntime

#endif