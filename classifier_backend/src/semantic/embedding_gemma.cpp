#include "classifier_backend/semantic/embedding_gemma.h"

#include "builtin/embedding_utils.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <thread>

using namespace rena;
using namespace rena::classifier_backend::semantic;

std::shared_ptr<embedding_gemma> embedding_gemma::create(const std::vector<core::category>& categories,
                                                         const llama_backend::config& config) {
    std::shared_ptr<embedding_gemma> p_instance(new embedding_gemma());
    if (!p_instance->_init(categories, config))
    {
        return nullptr;
    }
    return p_instance;
}

std::vector<core::classification_result> embedding_gemma::classify(const core::classification_input& input,
                                                                   core::common::span<int> candidate_category_ids) {
    const std::string& text = input.text;
    auto query_emb = this->_encode_text("task: classification | query: " + text);
    if (query_emb.empty())
    {
        return {};
    } // failed to encode text

    std::vector<std::pair<int, double>> raw_scores;
    raw_scores.reserve(this->_category_embds.size());
    for (const auto& [it_id, it_emb] : this->_category_embds)
    {
        raw_scores.emplace_back(it_id, builtin::embedding_utils::dot_product(query_emb, it_emb));
    }

    const double tau = (this->_temperature > 0) ? this->_temperature : 0.05;
    constexpr double bg_similarity = -1.0;
    return builtin::embedding_utils::temperature_softmax(raw_scores, tau, bg_similarity, candidate_category_ids);
}

embedding_gemma::~embedding_gemma() {
    llama_free(this->_ctx);
    llama_model_free(this->_model);
    return;
}

std::string_view embedding_gemma::get_model_name() const noexcept { return this->_model_name; }

core::common::span<core::lang> embedding_gemma::get_supported_langs() const noexcept {
    static constexpr core::lang supported_langs[] = { core::lang::UNIVERSAL };
    return supported_langs;
}

bool embedding_gemma::_init(const std::vector<core::category>& categories, const llama_backend::config& config) {
    if (!llama_backend::is_inited())
    {
        return false;
    } // llama backend not inited

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = (config.use_mmap) ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    mparams.n_gpu_layers = 0; // CPU only

    this->_model = llama_model_load_from_file(config.model_path.string().c_str(), mparams);
    if (!this->_model)
    {
        return false;
    } // failed to load model

    this->_temperature = config.temperature;
    this->_vocab = llama_model_get_vocab(this->_model);
    this->_n_embd = llama_model_n_embd(this->_model);

    char meta_buf[128] = {};
    if (llama_model_meta_val_str(this->_model, "general.name", meta_buf, sizeof(meta_buf)) > 0)
    {
        this->_model_name = meta_buf;
    } // get model name, set if exists

    llama_context_params cparams = llama_context_default_params();
    const uint32_t n_ctx_train = static_cast<uint32_t>(llama_model_n_ctx_train(this->_model)); // 2048
    constexpr uint32_t default_n_ctx = 512; // prevent OOM if config.n_ctx is not set
    const uint32_t requested_n_ctx = (config.n_ctx > 0) ? static_cast<uint32_t>(config.n_ctx) : default_n_ctx;
    const uint32_t n_ctx = std::min(requested_n_ctx, n_ctx_train);
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = n_ctx;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    const int32_t default_threads = std::max<int32_t>(1, static_cast<int32_t>(std::thread::hardware_concurrency() / 2));
    const int32_t n_threads = (config.n_threads > 0) ? config.n_threads : default_threads;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    this->_ctx = llama_init_from_model(this->_model, cparams);
    if (!this->_ctx)
    {
        return false;
    } // failed to create ctx

    this->_category_embds.clear();
    this->_category_embds.reserve(categories.size());
    for (const auto& it : categories)
    {
        const std::string& desc = it.prompt.empty() ? it.description : it.prompt;
        std::string title = it.name.empty() ? "none" : it.name;
        // std::string text = "title: " + title + " | text: " + desc;
        std::string text = std::string("title: ").append(title).append(" | text: ").append(desc);
        auto emb = this->_encode_text(text);
        if (emb.empty())
        {
            return false;
        }
        this->_category_embds.emplace_back(it.id, std::move(emb));
    }
    return true;
}

std::vector<float> embedding_gemma::_encode_text(std::string_view text) {
    if (!this->_ctx || !this->_vocab)
    {
        return {};
    }

    int32_t n_tokens =
        -llama_tokenize(this->_vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, true, false);
    if (n_tokens <= 0)
    {
        return {};
    }
    // calc needed tokens num

    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(this->_vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), n_tokens,
                              true, false);
    if (n_tokens < 0)
    {
        return {};
    } // failed to tokenize

    const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(this->_ctx));
    if (n_tokens > n_ctx)
    {
        llama_token last_token = tokens.back();
        tokens.resize(n_ctx);
        tokens.back() = last_token;
        // save EOS/SEP token
        n_tokens = n_ctx;
    } // cut tokens to fit ctx limit

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; i++)
    {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }

    llama_memory_clear(llama_get_memory(this->_ctx), true);

    int32_t rc = llama_decode(this->_ctx, batch);
    llama_batch_free(batch);
    if (rc != 0)
    {
        return {};
    } // decode failed

    const float* p_emb = llama_get_embeddings_seq(this->_ctx, 0);
    if (!p_emb)
    {
        return {};
    }
    // EmbeddingGemma 支持 Matryoshka 降维（768 / 512 / 256 / 128）：
    // 若想进一步省内存，可在这里直接截断到前 256 维后再做 l2_normalize
    std::vector<float> result(p_emb, p_emb + this->_n_embd);
    builtin::embedding_utils::l2_normalize(result);

    return result;
}
