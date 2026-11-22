#ifndef RENA_CLASSIFIER_BACKEND_SEMANTIC_BUILTIN_EMBEDDING_UTILS_H
#define RENA_CLASSIFIER_BACKEND_SEMANTIC_BUILTIN_EMBEDDING_UTILS_H

#include <vector>

namespace rena::classifier_backend::semantic::builtin::embedding_utils {

    void l2_normalize(std::vector<float>& vec);
    double dot_product(const std::vector<float>& a, const std::vector<float>& b);

} // namespace rena::classifier_backend::semantic::builtin::embedding_utils

#endif // RENA_CLASSIFIER_BACKEND_SEMANTIC_BUILTIN_EMBEDDING_UTILS_H
