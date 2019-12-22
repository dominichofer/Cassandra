#pragma once
#include "Core/Position.h"
#include <cstdint>
#include <functional>

uint64_t Pow_int(uint64_t base, uint64_t exponent);

int FullIndex(Position, BitBoard pattern);

int ReducedIndex(Position, BitBoard pattern_part);

void For_each_config(BitBoard pattern, const std::function<void(Position)>&);