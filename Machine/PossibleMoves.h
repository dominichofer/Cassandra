#pragma once
#include "MacrosHell.h"
#include <cstdint>

uint64_t PossibleMoves(uint64_t P, uint64_t O);

#if defined(HAS_AVX512)
	uint64_t PossibleMoves_AVX512(uint64_t P, uint64_t O);
#endif
#if defined(HAS_AVX2)
	uint64_t PossibleMoves_AVX2(uint64_t P, uint64_t O);
#endif
#if defined(HAS_SSE2)
	uint64_t PossibleMoves_SSE2(uint64_t P, uint64_t O);
#endif

uint64_t PossibleMoves_x64(uint64_t P, uint64_t O);