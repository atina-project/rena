#pragma once

#ifndef RENA_CLASSIFIER_BACKEND_SEMANTIC_MULTILINGUAL_E5_H
#define RENA_CLASSIFIER_BACKEND_SEMANTIC_MULTILINGUAL_E5_H

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "classifier_backend/semantic/llama_backend.h"
#include "core/classifier_interface.h"
#include "core/common/span.h"
#include "core/types.h"

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace rena::classifier_backend::semantic {

    class multilingual_e5 : public core::classifier_interface {
        public:
            static std::shared_ptr<multilingual_e5> create(const std::vector<core::category>& categories,
                                                           const llama_backend::config& config);

        public:
            ~multilingual_e5() override;
            multilingual_e5(const multilingual_e5&) = delete;
            multilingual_e5& operator=(const multilingual_e5&) = delete;
            multilingual_e5(multilingual_e5&&) = delete;
            multilingual_e5& operator=(multilingual_e5&&) = delete;

            std::vector<core::classification_result> classify(const core::classification_input& input,
                                                              core::common::span<int> candidate_category_ids = {}) override;

            std::string_view get_model_name() const noexcept override;
            core::common::span<core::lang> get_supported_langs() const noexcept override;

        private:
            multilingual_e5() = default;

            bool _init(const std::vector<core::category>& categories, const llama_backend::config& config);
            std::vector<float> _encode_text(std::string_view text);

            std::string _model_name = "multilingual-e5";

            llama_model* _model = nullptr;
            llama_context* _ctx = nullptr;
            const llama_vocab* _vocab = nullptr;
            int32_t _n_embd = 0;
            float _temperature = 0.01;

            std::vector<std::pair<int, std::vector<float>>> _category_embds;
    };

} // namespace rena::classifier_backend::semantic

#endif // RENA_CLASSIFIER_BACKEND_SEMANTIC_MULTILINGUAL_E5_H
