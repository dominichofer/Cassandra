#pragma once
#include "Score.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>

class ConfidenceLevel
{
	uint8_t value = 0;
public:
	ConfidenceLevel() noexcept = default;
	ConfidenceLevel(float value) noexcept : value(static_cast<uint8_t>(std::floor(value * 10.0f))) {}
	operator float() const { return value / 10.0f; }

	bool IsInfinit() const;
};

static ConfidenceLevel inf{ 25.5f };
