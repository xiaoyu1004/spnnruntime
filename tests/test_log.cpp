#include "log.h"

static int testLogMacro()
{
    spnnruntime::LoggerMgr::GetInstance()->setLevel(spnnruntime::Loglevel::DEBUG);
    spnnruntime::LoggerMgr::GetInstance()->init();

    for (int i = 0; i < 1; ++i)
    {
        SPNN_LOG_DEBUG(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_DEBUG(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_DEBUG(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_INFO(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_WARN(SPNN_ROOT_LOGGER) << "hello spnnruntime log macro test!";
        SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "hello spnn root logger!";
        SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "hello spnn root logger!";
        SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "hello spnn root logger!";
        SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "hello spnn root logger!";
        // SPNN_LOG_FATAL(SPNN_LOGGER("root")) << "hello 111111111111111111111111111111111111111111111spnn root logger!";
        // SPNN_LOG_FATAL(SPNN_LOGGER("root")) << "hello spnn root logger!";
        // SPNN_LOG_FATAL(SPNN_LOGGER("root")) << "hello 111111111111111111111111111111111spnn root logger!";
        // SPNN_LOG_FATAL(SPNN_LOGGER("root")) << "hello spnn root logger!";
        // SPNN_LOG_FATAL(SPNN_LOGGER("root")) << "hello spnn root logger!";
        // SPNN_LOG_FATAL(SPNN_LOGGER("root")) << "hello spnn root logger!";
    }
    return 0;
}

int main()
{
    return 0 || testLogMacro();
}