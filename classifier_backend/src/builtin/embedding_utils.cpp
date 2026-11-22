#include "builtin/embedding_utils.h"

#include <algorithm>
#include <cmath>

using namespace rena;
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

std::vector<core::classification_result>
embedding_utils::temperature_softmax(const std::vector<std::pair<int, double>>& raw_scores, double temperature,
                                     double bg_score, core::common::span<int> category_filter) {
    if (raw_scores.empty() || temperature <= 0.0)
    {
        return {};
    }

    double max_score = raw_scores.front().second;
    for (const auto& [_, score] : raw_scores)
    {
        max_score = std::max(max_score, score);
    }
    if (bg_score > -1.0)
    {
        max_score = std::max(max_score, bg_score);
    }

    double denom = (bg_score > -1.0) ? std::exp((bg_score - max_score) / temperature) : 0.0;
    for (const auto& [_, it_score] : raw_scores)
    {
        denom += std::exp((it_score - max_score) / temperature);
    }

    std::vector<core::classification_result> results;
    results.reserve(raw_scores.size());
    for (const auto& [it_id, it_raw_score] : raw_scores)
    {
        if (!category_filter.empty() && !category_filter.contains(it_id))
        {
            continue;
        }
        double score = std::exp((it_raw_score - max_score) / temperature) / denom;
        results.push_back({ it_id, score, it_raw_score });
    }

    return results;
}
