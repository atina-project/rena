#include "classifier_backend/semantic/llama_backend.h"

#include "core/log.h"
#include "llama.h"

using namespace rena;
using namespace rena::classifier_backend::semantic;

void llama_log_capture(ggml_log_level level, const char* text, void*) {
    if (!text)
    {
        return;
    }
    core::log::log_level mapped_level;
    switch (level)
    {
        case GGML_LOG_LEVEL_ERROR:
            mapped_level = core::log::log_level::ERROR;
            break;
        case GGML_LOG_LEVEL_WARN:
            mapped_level = core::log::log_level::WARNING;
            break;
        case GGML_LOG_LEVEL_INFO:
            mapped_level = core::log::log_level::DEBUG;
            break;
        case GGML_LOG_LEVEL_DEBUG:
            mapped_level = core::log::log_level::TRACE;
            break;
        case GGML_LOG_LEVEL_CONT:
        case GGML_LOG_LEVEL_NONE:
        default:
            return;
    }
    if (!rena::core::log::check_log_level(mapped_level))
    {
        return;
    }
    std::string_view sv(text);
    while (!sv.empty() && (sv.back() == '\n' || sv.back() == '\r'))
    {
        sv.remove_suffix(1);
    }
    if (sv.empty())
    {
        return;
    }
    rena::core::log::write_log(mapped_level, "llama", { "llama.cpp", 0, "" }, "%.*s", static_cast<int>(sv.size()),
                               sv.data());
    return;
}

bool llama_backend::inited = false;

void llama_backend::init() {
    if (!inited)
    {
        llama_log_set(&llama_log_capture, nullptr);
        llama_backend_init();
        inited = true;
    }
    return;
}

bool llama_backend::is_inited() { return inited; }
