#ifndef RENA_CLASSIFIER_BACKEND_SEMANTIC_EMBEDDING_GEMMA_H
#define RENA_CLASSIFIER_BACKEND_SEMANTIC_EMBEDDING_GEMMA_H

#include <memory>
#include <vector>

#include "classifier_backend/semantic/llama_backend.h"
#include "core/classifier_interface.h"
#include "core/common/span.h"
#include "core/types.h"

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace rena::classifier_backend::semantic {

    class embedding_gemma : public core::classifier_interface {

        public:
            static std::shared_ptr<embedding_gemma> create(const std::vector<core::category>& categories,
                                                           const llama_backend::config& config);

        public:
            ~embedding_gemma() override;
            embedding_gemma(const embedding_gemma&) = delete;
            embedding_gemma& operator=(const embedding_gemma&) = delete;
            embedding_gemma(embedding_gemma&&) = delete;
            embedding_gemma& operator=(embedding_gemma&&) = delete;

            std::vector<core::classification_result>
            classify(const core::classification_input& input,
                     core::common::span<int> candidate_category_ids = {}) override;

            std::string_view get_model_name() const noexcept override;
            core::common::span<core::lang> get_supported_langs() const noexcept override;

        private:
            embedding_gemma() = default;

            bool _init(const std::vector<core::category>& categories, const llama_backend::config& config);
            std::vector<float> _encode_text(std::string_view text);

            std::string _model_name = "embeddinggemma";
            llama_model* _model = nullptr;
            llama_context* _ctx = nullptr;
            const llama_vocab* _vocab = nullptr;
            int32_t _n_embd = 0;
            float _temperature = 0.05;

            std::vector<std::pair<int, std::vector<float>>> _category_embds;
    }; // class embedding_gemma

} // namespace rena::classifier_backend::semantic

#endif // RENA_CLASSIFIER_BACKEND_SEMANTIC_EMBEDDING_GEMMA_H
