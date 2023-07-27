#pragma once
#include "Core/Core.h"
#include "Position.h"

CUDA_CALLABLE Moves PossibleMoves(const Position&) noexcept;

namespace detail
{
	#ifdef __AVX512F__
		Moves PossibleMoves_AVX512(const Position&) noexcept;
	#endif

	#ifdef __AVX2__
		Moves PossibleMoves_AVX2(const Position&) noexcept;
	#endif

	CUDA_CALLABLE Moves PossibleMoves_x64(const Position&) noexcept;
}
