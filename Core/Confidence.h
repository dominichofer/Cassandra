#pragma once
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

class Confidence
{
	float z;
public:
	constexpr explicit Confidence(float z) noexcept : z(z) { assert(z >= 0); }

	[[nodiscard]] static constexpr Confidence Certain() noexcept { return Confidence{std::numeric_limits<decltype(z)>::infinity()}; }

	[[nodiscard]] auto operator<=>(const Confidence& o) const noexcept = default;

	[[nodiscard]] float sigmas() const noexcept { return z; }
	[[nodiscard]] bool IsCertain() const noexcept { return *this == Certain(); }
	[[nodiscard]] std::string to_string() const { return std::to_string(static_cast<int>(std::floor(100.0 * std::erf(z / std::sqrt(2.0))))) + '%'; }
};

[[nodiscard]] inline std::string to_string(const Confidence& c) { return c.to_string(); }
inline std::ostream& operator<<(std::ostream& os, const Confidence& c) { return os << to_string(c); }

[[nodiscard]] constexpr Confidence operator""_sigmas(long double z) { return Confidence(z); }