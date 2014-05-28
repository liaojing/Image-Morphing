#include <stdio.h>
#include <stdarg.h>

#include "error.h"

namespace nehab
{

void error_output(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

}
