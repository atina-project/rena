#include "classifier_backend/semantic/llama_backend.h"

#include "llama.h"

using namespace rena::classifier_backend::semantic;

bool llama_backend::inited = false;

void llama_backend::init() {
    if (!inited)
    {
        llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
        llama_backend_init();
        inited = true;
    }
    return;
}

bool llama_backend::is_inited() { return inited; }
