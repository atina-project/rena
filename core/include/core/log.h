#pragma once

#ifndef RENA_CORE_LOG_H
#define RENA_CORE_LOG_H

#include <chrono>
#include <cstdint>
#include <string_view>

#ifndef RENA_MIN_LOG_LEVEL
#ifdef NDEBUG
#define RENA_MIN_LOG_LEVEL INFO
#else // NDEBUG
#define RENA_MIN_LOG_LEVEL TRACE
#endif // NDEBUG
#endif // RENA_MIN_LOG_LEVEL

/**
 * write_log() is NOT thread-safe.
 */
#define RENA_LOG(level, tag, ...)                                                                              \
    do                                                                                                         \
    {                                                                                                          \
        if constexpr (::rena::core::log::log_level::level >= ::rena::core::log::log_level::RENA_MIN_LOG_LEVEL) \
        {                                                                                                      \
            if (::rena::core::log::check_log_level(::rena::core::log::log_level::level))                       \
            {                                                                                                  \
                ::rena::core::log::write_log(::rena::core::log::log_level::level, tag,                         \
                                             { __FILE__, __LINE__, __func__ }, __VA_ARGS__);                   \
            }                                                                                                  \
        }                                                                                                      \
    } while (0)

namespace rena::core::log {

    enum class log_level : uint8_t { TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL };

    // clang-format off

    struct log_location {
        const char* file;
        int line;
        const char* func;
    }; // struct log_location

    struct log_data {
        log_level level;
        std::string_view tag;
        std::string_view msg;
        log_location location;
        std::chrono::system_clock::time_point tp;
    }; // struct log_data

    // clang-format on

    using log_callback = void (*)(const log_data& log_data);

    void set_min_log_level(log_level level) noexcept;
    void set_log_callback(log_callback cb) noexcept;

    bool check_log_level(log_level level) noexcept;
    void write_log(log_level level, std::string_view tag, const log_location& location, const char* fstr, ...);

} // namespace rena::core::log

#endif // RENA_CORE_LOG_H
