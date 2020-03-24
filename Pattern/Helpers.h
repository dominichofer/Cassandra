#pragma once
#include "Core/Position.h"
#include <cstdint>
#include <functional>

uint64_t Pow_int(uint64_t base, uint64_t exponent);

int Index(const Position&, BitBoard pattern);

void For_each_config(BitBoard pattern, const std::function<void(Position)>&);