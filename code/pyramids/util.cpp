#include <cstring>
#include <cstdlib>
#include "util.h"

namespace util {
    int match(const char *argument, const char *option, const char **value) {
        int length = strlen(option);
        if (option[length-1] == ':') length--;
        if (strncmp(argument, option, length) == 0 
            && (argument[length] == ':' || argument[length] == '(' 
                || !argument[length])) {
            if (value) {
                if (option[length] == ':' && argument[length] == ':') length++;
                *value = argument+length;
            }
            return 1;
        } else return 0;
    }

    char *strdup(const char *str) {
        char *copy = (char *) malloc(strlen(str)+1);
        if (!copy) return NULL;
        return strcpy(copy, str);
    }
}
