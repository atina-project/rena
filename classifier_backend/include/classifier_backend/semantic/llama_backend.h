#pragma once

#ifndef RENA_CLASSIFIER_BACKEND_SEMANTIC_LLAMA_BACKEND_H
#define RENA_CLASSIFIER_BACKEND_SEMANTIC_LLAMA_BACKEND_H

#include <filesystem>

namespace rena::classifier_backend::semantic {

    class llama_backend {
        public:
            // clang-format off

            struct config {
                std::filesystem::path model_path;
                int n_threads = 0;
                int n_ctx = 0;
                bool use_mmap = true;
                float temperature = 0.0f;
            }; // struct config

            // clang-format on

        public:
            static void init();
            static bool is_inited();

        private:
            static bool inited;
    }; // class llama_backend

} // namespace rena::classifier_backend::semantic

#endif // RENA_CLASSIFIER_BACKEND_SEMANTIC_LLAMA_BACKEND_H
