#pragma once
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

class ConfidenceLevel
{
	float z;
public:
	constexpr explicit ConfidenceLevel(float z) noexcept : z(z) { assert(z >= 0); }

	[[nodiscard]] static constexpr ConfidenceLevel Certain() noexcept { return ConfidenceLevel{std::numeric_limits<decltype(z)>::infinity()}; }

	[[nodiscard]] auto operator<=>(const ConfidenceLevel& o) const noexcept = default;

	[[nodiscard]] float sigmas() const noexcept { return z; }
	[[nodiscard]] bool IsCertain() const noexcept { return *this == Certain(); }
	[[nodiscard]] std::string to_string() const { return std::to_string(static_cast<int>(std::floor(100.0 * std::erf(z / std::sqrt(2.0))))) + '%'; }
};

[[nodiscard]] inline std::string to_string(const ConfidenceLevel& c) { return c.to_string(); }
inline std::ostream& operator<<(std::ostream& os, const ConfidenceLevel& c) { return os << to_string(c); }

[[nodiscard]] constexpr ConfidenceLevel operator""_sigmas(long double z) { return ConfidenceLevel(z); }