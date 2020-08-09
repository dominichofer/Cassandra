#pragma once
#include "Position.h"
#include "Moves.h"
#include <cstdint>

[[nodiscard]]
Moves PossibleMoves(const Position&) noexcept;

namespace detail
{
	#if defined(__AVX512F__)
		[[nodiscard]]
		Moves PossibleMoves_AVX512(const Position&) noexcept;
	#endif
	#if defined(__AVX2__)
		[[nodiscard]]
		Moves PossibleMoves_AVX2(const Position&) noexcept;
	#endif

	[[nodiscard]]
	Moves PossibleMoves_x64(const Position&) noexcept;
}
