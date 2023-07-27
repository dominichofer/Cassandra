#pragma once
#include "Core/Core.h"
#include "Field.h"
#include "Position.h"

CUDA_CALLABLE uint64_t Flips(const Position&, Field move) noexcept;