#ifndef ERROR_H
#define ERROR_H

#include <stdlib.h>

namespace nehab
{

void error_output(const char *fmt, ...);

#define warnf(args) do { \
    nehab::error_output("warning: %s: %d: ", __FILE__, __LINE__); \
    nehab::error_output args; \
    nehab::error_output("\n"); \
} while(0)

#define errorf(args) do { \
    nehab::error_output("error: %s: %d: ", __FILE__, __LINE__); \
    nehab::error_output args; \
    nehab::error_output("\n"); \
    exit(1); \
} while(0)

#define assertf(cond, args) do { \
    if (!(cond)) { \
        nehab::error_output("assert: %s: %d: ", __FILE__, __LINE__); \
        nehab::error_output args; \
        nehab::error_output("\n"); \
        exit(1); \
    } \
} while(0)

#ifndef ERROR_NDEBUG
#define debugf(args) do { \
    nehab::error_output("debug: %s: %d: ", __FILE__, __LINE__); \
    nehab::error_output args; \
    nehab::error_output("\n"); \
} while(0)
#else
#define debugf
#endif

}

#endif
