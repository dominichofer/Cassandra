#pragma once
#include "Base/Base.h"
#include "Field.h"
#include "Position.h"

CUDA_CALLABLE uint64_t Flips(const Position&, Field move) noexcept;

namespace detail
{
#ifdef __AVX2__
	uint64_t Flips_AVX2(const Position&, Field move) noexcept;
#endif

	CUDA_CALLABLE uint64_t Flips_x64(const Position&, Field move) noexcept;
}