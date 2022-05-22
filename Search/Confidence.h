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

	auto operator<=>(const Confidence&) const noexcept = default;

	float sigmas() const noexcept { return z; }
	bool IsCertain() const noexcept { return *this == Certain(); }
};

inline std::string to_string(Confidence c) { return std::to_string(static_cast<int>(std::floor(100.0 * (1 + std::erf(c.sigmas() / std::sqrt(2))) / 2))) + '%'; }

constexpr Confidence operator""_sigmas(long double z) { return Confidence{ static_cast<float>(z) }; }
