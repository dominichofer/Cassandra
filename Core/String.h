#pragma once
#include <chrono>
#include <string>

// HH:MM:SS.sss
std::string HH_MM_SS(std::chrono::duration<double>);

std::string ShortTimeString(std::chrono::duration<double> duration);

// Maps input to (..,'n', 'u', 'm', '', 'k', 'M', 'G',..)
std::string MetricPrefix(int magnitude_base_1000) noexcept(false);

std::size_t ParseBytes(const std::string& bytes) noexcept(false);

constexpr int64_t operator""_kB(unsigned long long v) noexcept { return v * 1024; }
constexpr int64_t operator""_MB(unsigned long long v) noexcept { return v * 1024 * 1024; }
constexpr int64_t operator""_GB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024; }
constexpr int64_t operator""_TB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024; }
constexpr int64_t operator""_EB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024; }
constexpr int64_t operator""_ZB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }
constexpr int64_t operator""_YB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }
