#pragma once
#include "Core/Core.h"
#include <string>

// Maps input to (..,'n', 'u', 'm', '', 'k', 'M', 'G',..)
char MetricPrefix(int magnitude_base_1000) noexcept(false);

std::size_t ParseBytes(const std::string& bytes) noexcept(false);

constexpr int64 operator""_kB(unsigned long long v) noexcept { return v * 1024; }
constexpr int64 operator""_MB(unsigned long long v) noexcept { return v * 1024 * 1024; }
constexpr int64 operator""_GB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024; }
constexpr int64 operator""_TB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024; }
constexpr int64 operator""_EB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024; }
constexpr int64 operator""_ZB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }
constexpr int64 operator""_YB(unsigned long long v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }
