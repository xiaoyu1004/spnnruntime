#include "log.h"
#include "SPNNDefine.h"

#include <iostream>
#include <algorithm>
#include <ctime>
#include <regex>
#include <tuple>
#include <vector>
#include <map>
#include <functional>
#include <cassert>

namespace spnnruntime
{
    std::thread::id GetThreadId()
    {
        return std::this_thread::get_id();
    }

    std::string Loglevel::toString(Level level)
    {
#define XX(name)                        \
    if (Loglevel::Level::name == level) \
        return #name;
        XX(DEBUG);
        XX(INFO);
        XX(WARN);
        XX(ERROR);
        XX(FATAL);
#undef XX
        return "UNKNOW";
    }

    EventWrapper::~EventWrapper()
    {
        m_logger->log(m_event);
        if (m_event->getLevel() >= Loglevel::ERROR)
        {
            std::cout << std::flush;
            if (m_event->getLevel() >= Loglevel::FATAL)
            {
                std::terminate();
            }
        }
    }

    std::string LogFormatter::format(const Logger::ptr logger, const LogEvent::ptr event)
    {
        std::stringstream ss;
        for (auto &item : m_items)
        {
            item->format(ss, logger, event);
        }
        return ss.str();
    }

    class DateTimeFormatItem : public LogFormatter::FormatItem
    {
    public:
        DateTimeFormatItem(std::string &&format = "%Y-%m-%d %H:%M:%S") noexcept : m_format(std::move(format))
        {
        }

    public:
        void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            std::tm stime;
            time_t t = static_cast<time_t>(event->getTimestamp());
#if __unix__
            localtime_r(&t, &stime);
#elif _MSC_VER
            localtime_s(&stime, &t);
#endif
            char str[64] = {0};
            strftime(str, sizeof(str), m_format.c_str(), &stime);
            os << str;
        }

    private:
        std::string m_format;
    };

    class LoggerNameFormatItem : public LogFormatter::FormatItem
    {
    public:
        LoggerNameFormatItem(std::string &&) {}

    public:
        void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << logger->getName();
        }
    };

    class LogLevelFormatItem : public LogFormatter::FormatItem
    {
    public:
        LogLevelFormatItem(std::string &&) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << Loglevel::toString(event->getLevel());
        }
    };

    class FilenameFormatItem : public LogFormatter::FormatItem
    {
    public:
        FilenameFormatItem(std::string &&) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << event->getFileName();
        }
    };

    class LineFormatItem : public LogFormatter::FormatItem
    {
    public:
        LineFormatItem(std::string &&) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << event->getLine();
        }
    };

    class MessageFormatItem : public LogFormatter::FormatItem
    {
    public:
        MessageFormatItem(std::string &&) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << event->getContent();
        }
    };

    class NewLineFormatItem : public LogFormatter::FormatItem
    {
    public:
        NewLineFormatItem(std::string &&) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << "\n";
        }
    };

    class ThreadIdFormatItem : public LogFormatter::FormatItem
    {
    public:
        ThreadIdFormatItem(std::string &&) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << event->getThreadId();
        }
    };

    class TabFormatItem : public LogFormatter::FormatItem
    {
    public:
        TabFormatItem(std::string &&) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << "\t";
        }
    };

    class StrFormatItem : public LogFormatter::FormatItem
    {
    public:
        StrFormatItem(std::string &&str = "") : m_str(std::move(str)) {}

    public:
        virtual void format(std::ostream &os, const Logger::ptr logger, const LogEvent::ptr event) override
        {
            os << m_str;
        }

    private:
        std::string m_str;
    };

    void LogFormatter::init()
    {
        // %d{%Y-%m-%d %H:%M:%S}%T%n%T[%p]%T%f%T%l%T%m%L
        std::vector<std::tuple<std::string, std::string>> parsedVec;
        std::regex reg("[dTnptflmL\\[\\]]{1}(\\{.*\\}){0,1}");
        for (std::sregex_iterator it = std::sregex_iterator(m_pattern.begin(), m_pattern.end(), reg), end = std::sregex_iterator(); it != end; ++it)
        {
            std::string ptn = std::move(it->str());
            int type = ptn.size() == 1 ? 0 : 1;
            size_t start = ptn.find("{");
            std::string fmt = type == 0 ? "" : ptn.substr(start + 1, ptn.find("}") - start - 1);
            parsedVec.push_back(std::move(std::make_tuple(type ? ptn.substr(0, 1) : ptn, fmt)));
        }
        SPNN_ASSERT(!parsedVec.empty(), "parsedVec is empty!");
        static std::map<std::string, std::function<FormatItem::ptr(std::string &&)>> funcs = {
#define XX(str, C)                                                                                                          \
    {                                                                                                                       \
#str, [](std::string && fmt) { return std::dynamic_pointer_cast<FormatItem>(std::make_shared<C>(std::move(fmt))); } \
    }
            XX(d, DateTimeFormatItem),
            XX(T, TabFormatItem),
            XX(n, LoggerNameFormatItem),
            XX(s, StrFormatItem),
            XX(p, LogLevelFormatItem),
            XX(t, ThreadIdFormatItem),
            XX(f, FilenameFormatItem),
            XX(l, LineFormatItem),
            XX(m, MessageFormatItem),
            XX(L, NewLineFormatItem)
#undef XX
        };
        for (auto it = parsedVec.begin(), end = parsedVec.end(); it != end; ++it)
        {
            auto funcIt = funcs.find(std::get<0>(*it));
            if (funcIt == funcs.end())
            {
                m_items.push_back(std::make_shared<StrFormatItem>(std::move(std::get<0>(*it))));
            }
            else
            {
                m_items.push_back(funcIt->second(std::move(std::get<1>(*it))));
            }
        }
    }

    void StdoutLogAppender::log(const std::shared_ptr<Logger> logger, const LogEvent::ptr event)
    {
        if (event->getLevel() >= m_level)
        {
            std::cout << m_formatter->format(logger, event);
        }
    }

    FileLogAppender::FileLogAppender(std::string file, Loglevel::Level level) : LogAppender(level), m_file(file)
    {
        reopen();
    }

    void FileLogAppender::log(const std::shared_ptr<Logger> logger, const LogEvent::ptr event)
    {
        if (event->getLevel() >= m_level)
        {
            m_fs << m_formatter->format(logger, event);
        }
    }

    void Logger::log(const LogEvent::ptr event)
    {
        if (event->getLevel() >= m_level)
        {
            for (auto item : m_appenders)
            {
                item->log(shared_from_this(), event);
            }
        }
    }

    void Logger::addAppender(LogAppender::ptr appender)
    {
        if (!appender->getFormatter())
        {
            appender->setFormatter(m_formatter);
        }
        m_appenders.push_back(appender);
    }

    void Logger::delAppender(LogAppender::ptr appender)
    {
        auto it = std::find(m_appenders.begin(), m_appenders.end(), appender);
        m_appenders.erase(it);
    }
} // spnnruntime