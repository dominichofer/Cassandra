#pragma once
#include "Board/Board.h"
#include <cstdint>
#include <string>
#include <string_view>

class Score
{
	int8_t value{};
public:
	constexpr Score() noexcept = default;
	constexpr Score(int value) noexcept : value(value) {}
	constexpr operator int8_t() const { return value; }

	static Score FromString(std::string_view);
	Score operator-() const noexcept { return -value; }
};

std::string to_string(Score);

inline constexpr Score min_score{ -32 };
inline constexpr Score max_score{ +32 };
inline constexpr Score inf_score{ +33 };
inline constexpr Score undefined_score{ +35 };

Score EndScore(const Position&) noexcept;
Score StabilityBasedMaxScore(const Position&);