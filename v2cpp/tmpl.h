// FACE LAW v2 templates (see tmpl.cpp).
#pragma once
#include "facelaw2.h"
#include <vector>

namespace fl2 {
int make_template(const uint8_t *data, int w, int h, int channels, std::vector<uint8_t> &out, Eyes &eyes, Box &box);
int make_templates_all(const uint8_t *data, int w, int h, int channels, std::vector<std::vector<uint8_t>> &outs, std::vector<Eyes> &eyes);
void failed_template(std::vector<uint8_t> &out);
bool valid_template(const std::vector<uint8_t> &t);
double match_templates(const std::vector<uint8_t> &verif, const std::vector<uint8_t> &enrol, bool symmetric, bool &ok);
}  // namespace fl2
