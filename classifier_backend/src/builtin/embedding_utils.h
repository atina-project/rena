#pragma once

#ifndef RENA_CLASSIFIER_BACKEND_SEMANTIC_BUILTIN_EMBEDDING_UTILS_H
#define RENA_CLASSIFIER_BACKEND_SEMANTIC_BUILTIN_EMBEDDING_UTILS_H

#include <utility>
#include <vector>

#include "core/common/span.h"
#include "core/types.h"

namespace rena::classifier_backend::semantic::builtin::embedding_utils {

    void l2_normalize(std::vector<float>& vec);
    double dot_product(const std::vector<float>& a, const std::vector<float>& b);

    std::vector<core::classification_result> temperature_softmax(const std::vector<std::pair<int, double>>& raw_scores,
                                                                 double temperature, double bg_score = -1.0,
                                                                 core::common::span<int> category_filter = {});

} // namespace rena::classifier_backend::semantic::builtin::embedding_utils

#endif // RENA_CLASSIFIER_BACKEND_SEMANTIC_BUILTIN_EMBEDDING_UTILS_H
