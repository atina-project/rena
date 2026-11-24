#include "core/log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

using namespace rena::core;

const char* log_level_to_str(log::log_level level) noexcept {
    switch (level)
    {
        case log::log_level::TRACE:
            return "TRACE   ";
        case log::log_level::DEBUG:
            return "DEBUG   ";
        case log::log_level::INFO:
            return "INFO    ";
        case log::log_level::WARNING:
            return "WARNING ";
        case log::log_level::ERROR:
            return "ERROR   ";
        case log::log_level::CRITICAL:
            return "CRITICAL";
        default:
            return "WTF???  ";
    }
}

void g_default_log_callback(const log::log_data& data) {
    static std::mutex mtx;

    const auto now = std::chrono::system_clock::to_time_t(data.tp);
    const auto now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(data.tp.time_since_epoch()).count() % 1000;

    std::tm local_tm = {};
    localtime_r(&now, &local_tm);

    char time_buf[32] = {};
    std::snprintf(time_buf, sizeof(time_buf), "%02d:%02d:%02d.%03ld", local_tm.tm_hour, local_tm.tm_min,
                  local_tm.tm_sec, now_ms);
    std::lock_guard<std::mutex> lock(mtx);
    std::fprintf(stderr, "[%s] %s [%.*s] %.*s (%s:%d)\n", time_buf, log_level_to_str(data.level),
                 static_cast<int>(data.tag.size()), data.tag.data(), static_cast<int>(data.msg.size()), data.msg.data(),
                 data.location.file, data.location.line);
    // string_view may not end with '\0', str length must be specificated

    return;
}

std::atomic<log::log_level> g_min_log_level = log::log_level::RENA_MIN_LOG_LEVEL;
std::atomic<log::log_callback> g_log_cb = &g_default_log_callback;

bool log::check_log_level(log::log_level level) noexcept {
    return level >= g_min_log_level.load(std::memory_order_relaxed);
}

void log::set_min_log_level(log::log_level level) noexcept { g_min_log_level.store(level, std::memory_order_relaxed); }

void log::set_log_callback(log::log_callback cb) noexcept { g_log_cb.store(cb, std::memory_order_release); }

void log::write_log(log::log_level level, std::string_view tag, const log_location& location, const char* fstr, ...) {
    const log_callback log_cb = g_log_cb.load(std::memory_order_acquire);
    if (!log_cb || !fstr)
    {
        return;
    } // no callback / empty fstr

    const auto now = std::chrono::system_clock::now();

    char buf[512];
    std::string heap_buf;
    std::string_view msg;

    va_list args;
    va_start(args, fstr);

    va_list args_copy;
    va_copy(args_copy, args);
    const int len = std::vsnprintf(buf, sizeof(buf), fstr, args);
    va_end(args);

    if (len < 0)
    {
        va_end(args_copy);
        return;
    } // failed to format log msg

    if (static_cast<std::size_t>(len) < sizeof(buf))
    {
        msg = std::string_view(buf, len);
    }
    else
    {
        heap_buf.resize(len);
        std::vsnprintf(heap_buf.data(), heap_buf.size() + 1, fstr, args_copy);
        msg = heap_buf;
    } // msg too long
    va_end(args_copy);

    log_cb({ level, tag, msg, location, now });
    return;
}
