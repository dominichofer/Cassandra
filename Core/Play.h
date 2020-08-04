#pragma once
#include "Position.h"

[[nodiscard]]
Position Play(const Position&, Field move, BitBoard flips);

[[nodiscard]]
Position Play(const Position&, Field move);

[[nodiscard]]
Position PlayPass(const Position&);