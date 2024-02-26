/*
 * Copyright © 2024 MOQI SINGAPORE PTE. LTD.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
 */

#pragma once

#include <exception>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <execinfo.h>
#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>

#include <SearchIndex/Config.h>
#ifdef MYSCALE_MODE
#    include <Common/CurrentMemoryTracker.h>
#endif

#define DIV_ROUND_UP(numerator, denominator) \
    (((numerator) + (denominator)-1) / (denominator))

/// @brief Static configurations used by SearchIndex.
struct SIConfiguration
{
    inline static int VLOG_LEVEL = 0;
    inline static uint64_t DISKANN_SECTOR_LEN = 4096;
    inline static std::vector<int32_t> SCANN_LEVEL_NUM_LEAF_NODES;

    // check if the current thread should abort
    using CheckAbortHandler = std::function<void(void)>;
    inline static thread_local CheckAbortHandler check_abort = nullptr;

    static void setCurrentThreadCheckAbortHandler(CheckAbortHandler handler)
    {
        check_abort = handler;
    }

    static void clearCurrentThreadCheckAbortHandler() { check_abort = nullptr; }

    static void currentThreadCheckAndAbort()
    {
        if (check_abort != nullptr)
            check_abort();
    }
};

#define SI_VLOG(level, FMT, ...) \
    if (level <= SIConfiguration::VLOG_LEVEL) \
    { \
        printf(FMT, __VA_ARGS__); \
        printf("\n"); \
    }

namespace Search
{

/// @brief Manage memory of ostream by AccessibleStringBuf, used in
/// protobuf serialization.
class AccessibleStringBuf : public std::stringbuf
{
public:
    /// @brief Access the internal buffer without copying.
    const char * get_internal_buffer() const { return pbase(); }

    /// @brief Return the size of the internal buffer.
    std::size_t get_internal_buffer_size() const { return pptr() - pbase(); }
};

/// @brief Compute hash value of a pair.
struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> & p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Combine the hashes using xor
        return h1 ^ h2;
    }
};

using idx_t = int64_t;
using json = nlohmann::json;

// Maintain consistent with ClickHouse ErrorCodes.cpp
enum class ErrorCode
{
    UNSUPPORTED_PARAMETER = 2,
    CANNOT_READ_FROM_ISTREAM = 23,
    CANNOT_WRITE_TO_OSTREAM = 24,
    ATTEMPT_TO_READ_AFTER_EOF = 32,
    BAD_ARGUMENTS = 36,
    NOT_IMPLEMENTED = 48,
    LOGICAL_ERROR = 49,
    CANNOT_OPEN_FILE = 76,
    CANNOT_CLOSE_FILE = 77,
    CANNOT_TRUNCATE_FILE = 88,
    CANNOT_ALLOCATE_MEMORY = 173,
    ABORTED = 236,
    CANNOT_MUNMAP = 239,
};

template <typename Enum>
std::string enumToString(Enum value)
{
    return std::string(magic_enum::enum_name<Enum>(value));
}

template <typename Enum>
bool findEnumByName(std::string name, Enum & value)
{
    for (auto it : magic_enum::enum_entries<Enum>())
    {
        if (name == it.second)
        {
            value = it.first;
            return true;
        }
    }
    return false;
}

/**
 * @brief Simple scope guard with C++17.
 *
 * Scope guard manages resources associated with a C++ scope.
 * https://stackoverflow.com/a/61242721/1248093
 */
template <typename F>
struct OnExit
{
    F func;
    OnExit(F && f) : func(std::forward<F>(f)) { }
    ~OnExit() { func(); }
};

template <typename F>
OnExit(F && frv) -> OnExit<F>;

/// Get current memory usage by examining /proc/self/status
std::string getRSSUsage();

/// Print current and peak memory usage
void printMemoryUsage(const std::string & header = "");

/// Convert a vector to a string
template <typename T>
std::string vectorToString(
    const std::vector<T> & vec,
    const char * delimiter = " ",
    const char * prefix = "",
    const char * suffix = "")
{
    std::ostringstream oss;
    oss << prefix;
    int k = 0;
    for (const auto & value : vec)
        oss << (k++ > 0 ? delimiter : "") << value;
    oss << suffix;
    return oss.str();
}


};

#ifdef MYSCALE_MODE
#    ifdef MYSCALE_WITH_UPGRADE
#        include <Common/logger_useful.h>
#    else
#        include <base/logger_useful.h>
#    endif

inline Poco::Logger * get_logger()
{
    static Poco::Logger * logger = nullptr;
    if (logger == nullptr)
        logger = &Poco::Logger::get("SearchIndex");
    return logger;
}
#    define SI_LOG_INFO(...) \
        LOG_IMPL( \
            get_logger(), \
            DB::LogsLevel::information, \
            Poco::Message::PRIO_INFORMATION, \
            __VA_ARGS__)
#    define SI_LOG_DEBUG(...) \
        LOG_IMPL( \
            get_logger(), \
            DB::LogsLevel::information, \
            Poco::Message::PRIO_DEBUG, \
            __VA_ARGS__)
#    define SI_LOG_WARNING(...) \
        LOG_IMPL( \
            get_logger(), \
            DB::LogsLevel::information, \
            Poco::Message::PRIO_WARNING, \
            __VA_ARGS__)
#    define SI_LOG_ERROR(...) \
        LOG_IMPL( \
            get_logger(), \
            DB::LogsLevel::information, \
            Poco::Message::PRIO_ERROR, \
            __VA_ARGS__)
#    define SI_LOG_FATAL(...) \
        LOG_IMPL( \
            get_logger(), \
            DB::LogsLevel::information, \
            Poco::Message::PRIO_FATAL, \
            __VA_ARGS__)
#else
// Standalone Mode

// we use spd log to be more compatible with ClickHouse logging
#    include <spdlog/spdlog.h>
#    define SI_LOG_INFO spdlog::info
#    define SI_LOG_DEBUG spdlog::debug
#    define SI_LOG_WARNING spdlog::warn
#    define SI_LOG_ERROR spdlog::error
#    define SI_LOG_FATAL spdlog::critical
#endif


/// @brief Class of exceptions thrown by SearchIndex.
class SearchIndexException : public std::exception
{
private:
    Search::ErrorCode code;
    std::string msg;

public:
    explicit SearchIndexException(
        Search::ErrorCode code_, const std::string & msg_) :
        code(code_)
    {
        std::string code_str(magic_enum::enum_name(code));
        msg = "Error(" + code_str + "): " + msg_;
    }

    SearchIndexException(
        Search::ErrorCode code_,
        const std::string & msg_,
        const char * funcName,
        const char * file,
        int line) :
        code(code_)
    {
        std::string code_str(magic_enum::enum_name(code));
        int size = snprintf(
            nullptr,
            0,
            "Error(%s) in %s at %s:%d: %s",
            code_str.c_str(),
            funcName,
            file,
            line,
            msg_.c_str());
        msg.resize(size + 1);
        snprintf(
            &msg[0],
            msg.size(),
            "Error(%s) in %s at %s:%d: %s",
            code_str.c_str(),
            funcName,
            file,
            line,
            msg_.c_str());
    }

    /// from std::exception
    const char * what() const noexcept override { return msg.c_str(); }
    int getCode() const { return static_cast<int>(code); }
};

// Error handling macros
#ifdef MYSCALE_MODE
// In MyScale, we don't print function name, file name and line number to avoid too much debug info
#    define SI_THROW_MSG(code, MSG) \
        do \
        { \
            throw SearchIndexException(code, MSG); \
        } while (false)

#    define SI_THROW_FMT(code, FMT, ...) \
        do \
        { \
            std::string fmt_str; \
            int str_size = snprintf(nullptr, 0, FMT, __VA_ARGS__); \
            fmt_str.resize(str_size + 1); \
            snprintf(&fmt_str[0], fmt_str.size(), FMT, __VA_ARGS__); \
            throw SearchIndexException(code, fmt_str); \
        } while (false)
#else
#    define SI_THROW_MSG(code, MSG) \
        do \
        { \
            throw SearchIndexException( \
                code, MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
        } while (false)

#    define SI_THROW_FMT(code, FMT, ...) \
        do \
        { \
            std::string fmt_str; \
            int str_size = snprintf(nullptr, 0, FMT, __VA_ARGS__); \
            fmt_str.resize(str_size + 1); \
            snprintf(&fmt_str[0], fmt_str.size(), FMT, __VA_ARGS__); \
            throw SearchIndexException( \
                code, fmt_str, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
        } while (false)
#endif


#define SI_THROW_IF_NOT(X, code) \
    do \
    { \
        if (!(X)) \
        { \
            SI_THROW_FMT(code, "'%s' failed", #X); \
        } \
    } while (false)

#define SI_THROW_IF_NOT_MSG(X, code, MSG) \
    do \
    { \
        if (!(X)) \
        { \
            SI_THROW_FMT(code, "'%s' failed: " MSG, #X); \
        } \
    } while (false)

#define SI_THROW_IF_NOT_FMT(X, code, FMT, ...) \
    do \
    { \
        if (!(X)) \
        { \
            SI_THROW_FMT(code, "'%s' failed: " FMT, #X, __VA_ARGS__); \
        } \
    } while (false)

#ifdef MYSCALE_MODE
inline void checkAvailableMemory(size_t size)
{
    CurrentMemoryTracker::alloc(size);
    CurrentMemoryTracker::free(size);
}
#else
inline void checkAvailableMemory(size_t size) { }
#endif

class MemoryUsageRecorder
{
public:
    struct MemoryUsageRecord
    {
        size_t total_usage = 0;
        size_t current_usage = 0;
    };

    size_t getCurrentMemoryUsage()
    {
        std::string rss_str = Search::getRSSUsage();
        std::smatch pieces_match;
        if (std::regex_match(
                rss_str, pieces_match, std::regex("VmRSS:\\s*([0-9]+)\\s*.*")))
        {
            if (pieces_match.size() < 2)
            {
                return 0;
            }
            return std::stoul(pieces_match[1].str());
        }
        return 0;
    }

    void appendUsageRecord(std::string name)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (named_memory_usage.find(name) != named_memory_usage.end())
        {
            name = name + "_2";
        }
        named_memory_usage[name] = MemoryUsageRecord{
            .total_usage = getCurrentMemoryUsage(),
        };
        if (!last_record_name.empty())
        {
            named_memory_usage[name].current_usage = getCurrentMemoryUsage()
                - named_memory_usage[last_record_name].total_usage;
            SI_LOG_INFO(
                "MemoryUsageRecorder::{}: {} {} kB",
                __func__,
                name,
                named_memory_usage[name].current_usage);
        }
        last_record_name = name;
    }

    std::string last_record_name;
    std::mutex mutex;
    std::map<std::string, MemoryUsageRecord> named_memory_usage;
};

static MemoryUsageRecorder mem_usage_recorder;

#ifdef USE_MEMORY_RECORDER
#    define RECORD_MEMORY_USAGE(name) \
        mem_usage_recorder.appendUsageRecord((name))
#else
#    define RECORD_MEMORY_USAGE(...)
#endif
