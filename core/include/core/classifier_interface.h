#pragma once

#ifndef RENA_CORE_CLASSIFIER_INTERFACE_H
#define RENA_CORE_CLASSIFIER_INTERFACE_H

#include <string_view>
#include <vector>

#include "core/common/span.h"
#include "core/types.h"

namespace rena::core {

    class classifier_interface {
        public:
            virtual ~classifier_interface() = default;

            virtual std::vector<classification_result> classify(const classification_input& input,
                                                                common::span<int> candidate_category_ids = {}) = 0;

            virtual std::string_view get_model_name() const noexcept = 0;
            virtual common::span<lang> get_supported_langs() const noexcept = 0;
    };

} // namespace rena::core

#endif // RENA_CORE_CLASSIFIER_INTERFACE_H
