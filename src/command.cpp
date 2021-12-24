#include "command.h"
#include "setting/params.h"
#include "log.h"

#define DATAWRITELENGTH 128

namespace spnnruntime
{
    Command::Command()
    {
#if USE_INTER
        m_control_fd = open_control("/dev/xdma0_user");
        m_control_base = mmap_control(m_control_fd, MAP_SIZE);
        m_fd_event = open_event("/dev/xdma0_events_0");
        m_fd_c2h = open("/dev/xdma0_c2h_0", O_RDWR | O_NONBLOCK);
        m_fd_h2c = open("/dev/xdma0_h2c_0", O_RDWR);

        int ret = posix_memalign((void **)&m_write_data, 4096, g_reportLen + 4096);
        if (ret != 0)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "posix memalign error!";
        }
#else
        int ret = posix_memalign((void **)&m_head, 4096, 10 + 4096);
        if (ret != 0)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "posix memalign error!";
        }
#endif
    }

    void Command::loadData(void *_ptr, size_t size)
    {
#if USE_INTER
        ssize_t nread = read(m_fd_c2h, _ptr, size);
#else
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ssize_t nread = read(m_fd, _ptr, size);
        close(m_fd);
#endif
        if (nread == -1)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "read inter data error!";
        }
    }

    std::uint32_t Command::getInterType()
    {
#if USE_INTER
        readEvent(m_fd_event);
        std::uint32_t val = readControl(m_control_base, 14 << 2);
        return val & 0x0F;
#else
        m_fd = open(getFilePath().c_str(), O_RDWR);
        ssize_t nread = read(m_fd, m_head, 10);
        if (nread == -1)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "read file error!";
        }
        lseek(m_fd, 0x00000000, SEEK_SET);
        return (std::uint32_t)m_head[8];
#endif
    }

    DataType Command::getDataType(std::uint32_t intertype) const
    {
        return intertype == 4 ? DataType::report : DataType::wave;
    }

    bool Command::sendLaunchCmd(const unsigned char *_ptr)
    {
#if USE_INTER
        usleep(10);
        // *(m_write_data + 0) = 0xAA; // Head
        // *(m_write_data + 1) = 0x55; // Head
        // *(m_write_data + 2) = 0x41; // Head
        // *(m_write_data + 3) = 0x42; //启动包
        // *(m_write_data + 4) = 0x80; //帧长度
        // *(m_write_data + 5) = 0x00;
        // *(m_write_data + 6) = 0x00;
        // *(m_write_data + 7) = 0x05;
        // *(m_write_data + 8) = 0x00;
        // *(m_write_data + 9) = 0x00;
        // *(m_write_data + 10) = 0x37;
        // *(m_write_data + 11) = 0x00;
        // *(m_write_data + 12) = 0x00;
        // *(m_write_data + 13) = 0x00;
        // *(m_write_data + 14) = 0x00;
        // *(m_write_data + 15) = 0x00;
        // *(m_write_data + 16) = 0x00;
        // *(m_write_data + 17) = 0x00;
        // *(m_write_data + 18) = 0x00;
        // *(m_write_data + 19) = 0x00;
        // *(m_write_data + 20) = 0x00;
        // *(m_write_data + 21) = 0x00;
        // *(m_write_data + 22) = 0x00;
        // *(m_write_data + 23) = 0x00;
        // *(m_write_data + 24) = 0x4F;

        // for (int i = 25; i < DATAWRITELENGTH; i++)
        // {
        //     *(m_write_data + i) = 0x00;
        // }
        // *(m_write_data + 124) = 0x5A; // rear
        // *(m_write_data + 125) = 0xF8; // rear
        // *(m_write_data + 126) = 0x5A; // rear
        // *(m_write_data + 127) = 0xF8; // rear

        lseek(m_fd_h2c, 0xC0000000, SEEK_SET);
        memcpy(m_write_data, _ptr, DATAWRITELENGTH);
        int status = write(m_fd_h2c, m_write_data, DATAWRITELENGTH);
        usleep(10);
        return status == -1 ? false : true;
#else
        return true;
#endif
    }

    bool Command::sendControlCmd(const unsigned char *_ptr)
    {
#if USE_INTER
        usleep(10);
        // *(m_write_data + 0) = 0xAA; // Head
        // *(m_write_data + 1) = 0x55; // Head
        // *(m_write_data + 2) = 0x41; // Head
        // *(m_write_data + 3) = 0x41; // Head
        // *(m_write_data + 4) = 0x80; //帧长度
        // *(m_write_data + 5) = 0x00;
        // *(m_write_data + 6) = 0x00;  //阵面号
        // *(m_write_data + 7) = 0x0a;  //包类型
        // *(m_write_data + 8) = 0x03;  //数据类型0
        // *(m_write_data + 9) = 0x01;  //数据类型1
        // *(m_write_data + 10) = 0x00; //数据类型2
        // *(m_write_data + 11) = 0x00; //数据类型3

        // *(m_write_data + 12) = 0x03;
        // *(m_write_data + 13) = 0x00;
        // *(m_write_data + 14) = 0x00;
        // *(m_write_data + 15) = 0x55;
        // *(m_write_data + 16) = 0x00;
        // *(m_write_data + 17) = 0x00;
        // *(m_write_data + 18) = 0x00;
        // *(m_write_data + 19) = 0x00;
        // *(m_write_data + 20) = 0x00;
        // *(m_write_data + 21) = 0x00;
        // *(m_write_data + 22) = 0x00;
        // *(m_write_data + 23) = 0x00;
        // *(m_write_data + 24) = 0x00;

        // for (int i = 25; i < DATAWRITELENGTH; i++)
        // {
        //     *(m_write_data + i) = 0x00;
        // }

        // *(m_write_data + 32) = 0xFF;
        // *(m_write_data + 33) = 0x7F;
        // *(m_write_data + 34) = 0x00;
        // *(m_write_data + 35) = 0x00;

        // *(m_write_data + 36) = 0xFF;
        // *(m_write_data + 37) = 0x7F;
        // *(m_write_data + 38) = 0x00;
        // *(m_write_data + 39) = 0x00;

        // *(m_write_data + 40) = 0xFF;
        // *(m_write_data + 41) = 0x7F;
        // *(m_write_data + 42) = 0x00;
        // *(m_write_data + 43) = 0x00;

        // *(m_write_data + 44) = 0xFF;
        // *(m_write_data + 45) = 0x7F;
        // *(m_write_data + 46) = 0x00;
        // *(m_write_data + 47) = 0x00;

        // *(m_write_data + 50) = 0xaa;  //环回测试
        // *(m_write_data + 51) = 0xaa;  //环回测试
        // *(m_write_data + 56) = 0x05;  //环回测试
        // *(m_write_data + 124) = 0x5A; //帧尾
        // *(m_write_data + 125) = 0xF8; //帧尾
        // *(m_write_data + 126) = 0x5A; //帧尾
        // *(m_write_data + 127) = 0xF8; //帧尾

        lseek(m_fd_h2c, 0xC0000000, SEEK_SET);
        memcpy(m_write_data, _ptr, DATAWRITELENGTH);
        int status = write(m_fd_h2c, m_write_data, DATAWRITELENGTH);
        usleep(10);
        return status == -1 ? false : true;
#else
        return true;
#endif
    }

    int Command::openEvent(const char *filename)
    {
        int fd = open(filename, O_RDWR | O_SYNC);
        if (fd == -1)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "open event fd error!";
        }
        return fd;
    }

    int Command::readEvent(int fd)
    {
        int val;
        int ret = read(fd, &val, 4);
        if (ret == -1)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "read event error!";
        }
        return val;
    }

    int Command::openControl(const char *filename)
    {
        int fd = open(filename, O_RDWR | O_SYNC);
        if (fd == -1)
        {
            SPNN_LOG_FATAL(SPNN_ROOT_LOGGER) << "open control error";
        }
        return fd;
    }

    void *Command::mmapControl(int fd, long mapsize)
    {
        void *vir_addr;
        vir_addr = mmap(0, mapsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        return vir_addr;
    }

    void Command::writeControl(void *base_addr, int offset, uint32_t val)
    {
        uint32_t writeval = htoll(val);
        *((uint32_t *)(base_addr) + offset) = writeval;
    }

    uint32_t Command::readControl(void *base_addr, int offset)
    {
        uint32_t read_result = *((uint32_t *)((unsigned char *)base_addr + offset));
        read_result = ltohl(read_result);
        return read_result;
    }

    std::string Command::getFilePath()
    {
        return "/home/yunjian/projects/spnnruntime/dataset/received593_1.bin";
    }
} // spnnruntime