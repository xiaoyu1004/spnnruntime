#ifndef LOG_H
#define LOG_H

#include "utils/common.h"
#include "SPNNDefine.h"
#include "singleton.hpp"

#include <cstdint>
#include <sstream>
#include <memory>
#include <list>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>

#define SPNN_LOGGER(...) spnnruntime::LoggerMgr::GetInstance()->getLogger(__VA_ARGS__)
#define SPNN_ROOT_LOGGER spnnruntime::LoggerMgr::GetInstance()->getLogger("root")

#define SPNN_LOG_LEVEL(logger, level) \
    if (level >= logger->getLevel())  \
    spnnruntime::EventWrapper(logger, spnnruntime::LogEvent::ptr(new spnnruntime::LogEvent(__FILE__, __LINE__, level, spnnruntime::GetThreadId(), time(NULL)))).getSS()

#define SPNN_LOG_DEBUG(logger) SPNN_LOG_LEVEL(logger, spnnruntime::Loglevel::DEBUG)
#define SPNN_LOG_INFO(logger) SPNN_LOG_LEVEL(logger, spnnruntime::Loglevel::INFO)
#define SPNN_LOG_WARN(logger) SPNN_LOG_LEVEL(logger, spnnruntime::Loglevel::WARN)
#define SPNN_LOG_ERROR(logger) SPNN_LOG_LEVEL(logger, spnnruntime::Loglevel::ERROR)
#define SPNN_LOG_FATAL(logger) SPNN_LOG_LEVEL(logger, spnnruntime::Loglevel::FATAL)

#define CHECK(a) \
    if (!(a))    \
    SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "CHECK FAILED(" << #a << " = " << (a) << ") "

#define CHECK_BINARY_OP(name, op, a, b) \
    if (!((a)op(b)))                    \
    SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "CHECK" << #name << " FAILED(" << #a << " " << #op << " " << #b << " vs. " << (a) << " " << #op << " " << (b) << ") "

#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)
#define CHECK_NOTNULL(x) \
    ((x) == NULL ? SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "Check  notnull: " #x << ' ', (x) : (x))

namespace spnnruntime
{
    std::thread::id GetThreadId();

    class Logger;

    class Loglevel
    {
    public:
        enum Level
        {
            UNKNOW = -1,
            DEBUG = 0,
            INFO = 1,
            WARN = 2,
            ERROR = 3,
            FATAL = 4
        };

        static std::string toString(Level level);
    };

    class LogEvent
    {
    public:
        typedef std::shared_ptr<LogEvent> ptr;
        LogEvent(const char *filename, std::uint32_t line, Loglevel::Level level, std::thread::id threadId, std::uint64_t timestamp) noexcept
            : m_filename(filename),
              m_line(line),
              m_level(level),
              m_threadId(threadId),
              m_timestamp(timestamp)
        {
        }

        ~LogEvent() {}

    public:
        const std::string &getFileName() const { return m_filename; }
        std::uint32_t getLine() const { return m_line; }
        Loglevel::Level getLevel() const { return m_level; }
        std::thread::id getThreadId() const { return m_threadId; }
        std::uint64_t getTimestamp() const { return m_timestamp; }
        std::string getContent() const { return m_content.str(); }
        std::stringstream &getSS() { return m_content; }

    private:
        DISABLE_COPY_AND_ASSIGN(LogEvent)
        std::string m_filename;
        std::uint32_t m_line;
        Loglevel::Level m_level;
        std::thread::id m_threadId;
        std::uint64_t m_timestamp;
        std::stringstream m_content;
    };

    class EventWrapper
    {
    public:
        EventWrapper(std::shared_ptr<Logger> logger, LogEvent::ptr event) : m_logger(logger), m_event(event)
        {
        }

        ~EventWrapper();

    public:
        std::stringstream &getSS()
        {
            return m_event->getSS();
        }

    private:
        std::shared_ptr<Logger> m_logger;
        LogEvent::ptr m_event;
    };

    class LogFormatter
    {
    public:
        typedef std::shared_ptr<LogFormatter> ptr;
        LogFormatter(std::string &&pattern = "%d{%Y-%m-%d %H:%M:%S}%T[%n]%T[%p]%T%t%T%f%T%l%T%m%L") noexcept : m_pattern(std::move(pattern)) { init(); };

    public:
        std::string format(const std::shared_ptr<Logger> logger, const LogEvent::ptr event);

        class FormatItem
        {
        public:
            typedef std::shared_ptr<FormatItem> ptr;
            FormatItem() {}
            virtual ~FormatItem() {}

        public:
            virtual void format(std::ostream &os, const std::shared_ptr<Logger> logger, const LogEvent::ptr event) = 0;
            DISABLE_COPY_AND_ASSIGN(FormatItem)
        };

    private:
        void init();

    private:
        DISABLE_COPY_AND_ASSIGN(LogFormatter)
        std::string m_pattern;
        std::list<FormatItem::ptr> m_items;
    };

    class LogAppender
    {
    public:
        typedef std::shared_ptr<LogAppender> ptr;
        LogAppender(Loglevel::Level level) : m_level(level){};
        virtual ~LogAppender() {}

    public:
        virtual void log(const std::shared_ptr<Logger> logger, const LogEvent::ptr event) = 0;
        void setLevel(Loglevel::Level level) { m_level = level; }
        Loglevel::Level getLevel() const { return m_level; }
        void setFormatter(const LogFormatter::ptr formatter) { m_formatter = formatter; }
        LogFormatter::ptr getFormatter() const { return m_formatter; }

    protected:
        Loglevel::Level m_level;
        LogFormatter::ptr m_formatter;
        DISABLE_COPY_AND_ASSIGN(LogAppender)
    };

    class StdoutLogAppender : public LogAppender
    {
    public:
        typedef std::shared_ptr<StdoutLogAppender> ptr;
        StdoutLogAppender(Loglevel::Level level = Loglevel::INFO) : LogAppender(level) {}

    public:
        virtual void log(const std::shared_ptr<Logger> logger, const LogEvent::ptr event) override;
    };

    class FileLogAppender : public LogAppender
    {
    public:
        typedef std::shared_ptr<FileLogAppender> ptr;
        FileLogAppender(std::string file, Loglevel::Level level = Loglevel::INFO);

    public:
        virtual void log(const std::shared_ptr<Logger> logger, const LogEvent::ptr event) override;

    private:
        bool reopen()
        {
            if (!m_fs.is_open())
            {
                m_fs.open(m_file.c_str());
            }
            return !!m_fs;
        }

    private:
        std::string m_file;
        std::ofstream m_fs;
    };

    class Logger : public std::enable_shared_from_this<Logger>
    {
    public:
        typedef std::shared_ptr<Logger> ptr;
        Logger(std::string &name, Loglevel::Level level) : m_name(name), m_level(level)
        {
        }

    public:
        void log(const LogEvent::ptr event);
        std::string getName() const { return m_name; }
        Loglevel::Level getLevel() const { return m_level; }
        void setLevel(Loglevel::Level level) { m_level = level; }
        void setFormatter(const LogFormatter::ptr formatter) { m_formatter = formatter; };
        LogFormatter::ptr getFormatter() const { return m_formatter; }
        void addAppender(LogAppender::ptr appender);
        void delAppender(LogAppender::ptr appender);

    private:
        DISABLE_COPY_AND_ASSIGN(Logger)
        std::string m_name;
        Loglevel::Level m_level;
        LogFormatter::ptr m_formatter;
        std::list<LogAppender::ptr> m_appenders;
    };

    class LogManager
    {
    public:
        LogManager() : m_level(Loglevel::INFO), m_appender(std::make_shared<StdoutLogAppender>()), m_formatter(std::make_shared<LogFormatter>())
        {
        }

    public:
        typedef std::shared_ptr<LogManager> ptr;
        void setLevel(Loglevel::Level level) { m_level = level; }
        void setAppender(LogAppender::ptr appender) { m_appender = appender; }
        void setFormatter(LogFormatter::ptr formatter) { m_formatter = formatter; }

        void init()
        {
            createLogger("root");
        }

        Logger::ptr getLogger(const std::string &name = "root")
        {
            auto it = m_loggers.find(name);
            SPNN_ASSERT(it != m_loggers.end(), "cannot get logger whose name is %s!", name.c_str());
            return it->second;
        }

        void createLogger(std::string name, Loglevel::Level level = Loglevel::UNKNOW, std::vector<LogAppender::ptr> appenders = {}, LogFormatter::ptr formatter = nullptr)
        {
            std::for_each(name.begin(), name.end(), ::tolower);
            Logger::ptr logger = std::make_shared<Logger>(name, level == Loglevel::UNKNOW ? m_level : level);
            logger->setFormatter(formatter ? formatter : m_formatter);
            if (appenders.empty())
            {
                logger->addAppender(m_appender);
            }
            else
            {
                for (auto it = appenders.begin(), end = appenders.end(); it != end; ++it)
                {
                    logger->addAppender(*it);
                }
            }
            m_loggers[name] = logger;
        }

    private:
        DISABLE_COPY_AND_ASSIGN(LogManager)
        Loglevel::Level m_level;
        LogAppender::ptr m_appender;
        LogFormatter::ptr m_formatter;
        std::map<std::string, Logger::ptr> m_loggers;
    };

    typedef Singleton<LogManager> LoggerMgr;
} // spnnruntime

#endif