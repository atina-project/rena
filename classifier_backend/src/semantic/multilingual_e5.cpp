#include "classifier_backend/semantic/multilingual_e5.h"
#include "builtin/embedding_utils.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <thread>

using namespace rena;
using namespace rena::classifier_backend::semantic;

std::shared_ptr<multilingual_e5> multilingual_e5::create(const std::vector<core::category>& categories,
                                                         const llama_backend::config& config) {
    std::shared_ptr<multilingual_e5> p_instance(new multilingual_e5());
    if (!p_instance->_init(categories, config))
    {
        return nullptr;
    }
    return p_instance;
}

std::vector<core::classification_result> multilingual_e5::classify(const core::classification_input& input,
                                                                   core::common::span<int> candidate_category_ids) {
    const std::string& text = input.text;
    auto query_emb = this->_encode_text("passage: " + text);
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

    const double tau = (this->_temperature > 0.0f) ? this->_temperature : 0.01;
    constexpr double bg_similarity = 0.75;
    return builtin::embedding_utils::temperature_softmax(raw_scores, tau, bg_similarity, candidate_category_ids);
}

multilingual_e5::~multilingual_e5() {
    llama_free(this->_ctx);
    llama_model_free(this->_model);
    return;
}

std::string_view multilingual_e5::get_model_name() const noexcept { return this->_model_name; }

core::common::span<core::lang> multilingual_e5::get_supported_langs() const noexcept {
    static constexpr core::lang supported_langs[] = { core::lang::UNIVERSAL };
    return supported_langs;
}

bool multilingual_e5::_init(const std::vector<core::category>& categories, const llama_backend::config& config) {
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
    switch (this->_n_embd)
    {
        case 384:
            this->_model_name = "multilingual-e5-small";
            break;
        case 768:
            this->_model_name = "multilingual-e5-base";
            break;
        case 1024:
            this->_model_name = "multilingual-e5-large";
            break;
        default:
            this->_model_name = "multilingual-e5-unknown";
            break;
    } // set model name

    llama_context_params cparams = llama_context_default_params();
    uint32_t n_ctx = static_cast<uint32_t>(llama_model_n_ctx_train(this->_model));
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
        std::string text = "query: " + (it.prompt.empty() ? it.description : it.prompt);
        auto emb = this->_encode_text(text);
        if (emb.empty())
        {
            return false;
        } // failed to encode text
        this->_category_embds.emplace_back(it.id, std::move(emb));
    } // encode all categories

    return true;
}

std::vector<float> multilingual_e5::_encode_text(std::string_view text) {
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

    int32_t rc =
        llama_model_has_encoder(this->_model) ? llama_encode(this->_ctx, batch) : llama_decode(this->_ctx, batch);
    llama_batch_free(batch);
    if (rc != 0)
    {
        return {};
    } // encode / decode failed

    const float* p_emb = llama_get_embeddings_seq(this->_ctx, 0);
    if (!p_emb)
    {
        return {};
    }
    std::vector<float> result(p_emb, p_emb + this->_n_embd);
    builtin::embedding_utils::l2_normalize(result);

    return result;
}
