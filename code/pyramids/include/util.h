#ifndef UTIL_H
#define UTIL_H

namespace util {
    int match(const char *argument, const char *option, const char **value); 
    char *strdup(const char *str);
}

#endif
