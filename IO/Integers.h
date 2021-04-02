#pragma once
#include "Core/Core.h"
#include <string>

// Maps input to (.., "-1", "+0", "+1", ..)
std::string SignedInt(int);

// Maps input to (.., "-01", "+00", "+01", ..)
std::string DoubleDigitSignedInt(int);

// Maps input to (..,'n', 'u', 'm', '', 'k', 'M', 'G',..)
char MetricPrefix(int magnitude_base_1000) noexcept(false);

[[nodiscard]] std::size_t ParseBytes(const std::string& bytes) noexcept(false);

[[nodiscard]] constexpr int64 operator""_kB(uint64 v) noexcept { return v * 1024; }
[[nodiscard]] constexpr int64 operator""_MB(uint64 v) noexcept { return v * 1024 * 1024; }
[[nodiscard]] constexpr int64 operator""_GB(uint64 v) noexcept { return v * 1024 * 1024 * 1024; }
[[nodiscard]] constexpr int64 operator""_TB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024; }
[[nodiscard]] constexpr int64 operator""_EB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024; }
[[nodiscard]] constexpr int64 operator""_ZB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }
[[nodiscard]] constexpr int64 operator""_YB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }
