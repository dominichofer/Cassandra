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

	static constexpr Confidence Certain() noexcept { return Confidence{std::numeric_limits<decltype(z)>::infinity()}; }

	auto operator<=>(const Confidence& o) const noexcept = default;

	float sigmas() const noexcept { return z; }
	bool IsCertain() const noexcept { return *this == Certain(); }
	std::string to_string() const { return std::to_string(static_cast<int>(std::floor(100.0 * std::erf(z / std::sqrt(2.0))))) + '%'; }
};

inline std::string to_string(const Confidence& c) { return c.to_string(); }
inline std::ostream& operator<<(std::ostream& os, const Confidence& c) { return os << to_string(c); }

constexpr Confidence operator""_sigmas(long double z) { return Confidence{ static_cast<float>(z) }; }