#pragma once

#ifndef RENA_CLASSIFIER_BACKEND_BUILTIN_AC_AUTOMATON_INTERFACE_H
#define RENA_CLASSIFIER_BACKEND_BUILTIN_AC_AUTOMATON_INTERFACE_H

#include <cstddef>
#include <string_view>
#include <vector>

namespace rena::classifier_backend::builtin {

    template<class Trule>
    class ac_automaton_interface {
        public:
            // clang-format off

            struct pattern_data {
                std::string_view pattern; // pattern string
                Trule hit_rule;           // user-defined rule payload bound to each pattern (passed to hit_cb upon match)
            }; // struct pattern_data

            // clang-format on

            using pattern_hit_callback = void (*)(const Trule&, std::size_t begin_pos, std::size_t end_pos,
                                                  void* user_data);

        public:
            virtual ~ac_automaton_interface() = default;

            virtual bool compile(const std::vector<pattern_data>& patterns, bool case_insensitive = true) = 0;
            virtual void match(std::string_view text, pattern_hit_callback hit_cb, void* user_data = nullptr) const = 0;
            virtual void clear() noexcept = 0;

    }; // class ac_automaton_interface

} // namespace rena::classifier_backend::builtin

#endif // RENA_CLASSIFIER_BACKEND_BUILTIN_AC_AUTOMATON_INTERFACE_H
