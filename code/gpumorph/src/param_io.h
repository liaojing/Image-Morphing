#ifndef GPUMORPH_PARAM_IO_H
#define GPUMORPH_PARAM_IO_H

#include <string>

struct Parameters;

void parse_config_xml(Parameters &params, const std::string &fname);

#endif
