#pragma once
#include "ConfidenceLevel.h"
#include <cstdint>
#include <string>
#include <string_view>

class Intensity
{
public:
	int8_t depth;
	ConfidenceLevel level;

	Intensity(int8_t depth, float level = 25.5f) noexcept : depth(depth), level(level) {}
	static Intensity FromString(std::string_view);

	std::string to_string() const;

	bool operator==(const Intensity& o) const noexcept { return depth == o.depth and level == o.level; }
	bool operator!=(const Intensity& o) const noexcept { return not(*this == o); }
	bool operator<(const Intensity& o) const noexcept { return (depth < o.depth and level <= o.level) or (depth <= o.depth and level < o.level); }
	bool operator>(const Intensity& o) const noexcept { return (depth > o.depth and level >= o.level) or (depth >= o.depth and level > o.level); }
	bool operator<=(const Intensity& o) const noexcept { return depth <= o.depth and level <= o.level; }
	bool operator>=(const Intensity& o) const noexcept { return depth >= o.depth and level >= o.level; }


	Intensity operator+(int depth) const noexcept { return { static_cast<int8_t>(this->depth + depth), level }; }
	Intensity operator-(int depth) const noexcept { return { static_cast<int8_t>(this->depth - depth), level }; }

	bool IsExact() const noexcept { return level.IsInfinit(); }
};

std::string to_string(const Intensity&);
