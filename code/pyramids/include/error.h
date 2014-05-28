#ifndef ERROR_H
#define ERROR_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
#endif
void error_output(const char *fmt, ...);

#define warnf(args) do { \
    error_output("warning: %s: %d: ", __FILE__, __LINE__); \
    error_output args; \
    error_output("\n"); \
} while(0)

#define errorf(args) do { \
    error_output("error: %s: %d: ", __FILE__, __LINE__); \
    error_output args; \
    error_output("\n"); \
    exit(1); \
} while(0)

#define assertf(cond, args) do { \
    if (!(cond)) { \
        error_output("assert: %s: %d: ", __FILE__, __LINE__); \
        error_output args; \
        error_output("\n"); \
        exit(1); \
    } \
} while(0)

#ifndef ERROR_NDEBUG
#define debugf(args) do { \
    error_output("debug: %s: %d: ", __FILE__, __LINE__); \
    error_output args; \
    error_output("\n"); \
} while(0)
#else
#define debugf
#endif

#endif
