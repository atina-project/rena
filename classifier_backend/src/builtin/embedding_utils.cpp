#include "builtin/embedding_utils.h"

#include <cmath>

using namespace rena::classifier_backend::semantic::builtin;

void embedding_utils::l2_normalize(std::vector<float>& vec) {
    double sum_sq = 0.0;
    for (float v : vec)
    {
        sum_sq += static_cast<double>(v) * v;
    }
    double norm = std::sqrt(sum_sq);
    if (norm > 1e-12)
    {
        for (float& v : vec)
        {
            v = static_cast<float>(v / norm);
        }
    }
    return;
}

double embedding_utils::dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    std::size_t size = std::min(a.size(), b.size());
    for (size_t i = 0; i < size; ++i)
    {
        sum += static_cast<double>(a[i]) * b[i];
    }
    return sum;
}
