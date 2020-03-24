#pragma once
#include "MacrosHell.h"
#include <cstdint>

[[nodiscard]]
uint64_t PossibleMoves(uint64_t P, uint64_t O);

namespace detail
{
	#if defined(__AVX512F__)
		[[nodiscard]]
		uint64_t PossibleMoves_AVX512(uint64_t P, uint64_t O);
	#endif
	#if defined(__AVX2__)
		[[nodiscard]]
		uint64_t PossibleMoves_AVX2(uint64_t P, uint64_t O);
	#endif
	#if defined(__AVX2__)
		[[nodiscard]]
		uint64_t PossibleMoves_SSE2(uint64_t P, uint64_t O);
	#endif

	[[nodiscard]]
	uint64_t PossibleMoves_x64(uint64_t P, uint64_t O);
}