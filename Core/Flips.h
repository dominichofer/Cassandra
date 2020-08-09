#pragma once
#include "Position.h"

[[nodiscard]]
BitBoard Flips(const Position&, Field move) noexcept;
