#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include "SPNNDefine.h"

#define DISABLE_COPY_AND_ASSIGN(classname)            \
private:                                              \
    classname(const classname &) = delete;            \
    classname(classname &&) = delete;                 \
    classname &operator=(const classname &) = delete; \
    classname &operator=(classname &&) = delete;

#endif