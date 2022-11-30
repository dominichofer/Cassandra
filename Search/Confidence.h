#pragma once
#include "Core/Core.h"
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
	static constexpr Confidence Uncertain() noexcept { return Confidence{ 0 }; }

	auto operator<=>(const Confidence&) const noexcept = default;

	float sigmas() const noexcept { return z; }
	bool IsCertain() const noexcept { return *this == Certain(); }
};

inline std::string to_string(Confidence c) { return fmt::format("{:1.1f}", c.sigmas()); }

constexpr Confidence operator""_sigmas(long double z) { return Confidence{ static_cast<float>(z) }; }
