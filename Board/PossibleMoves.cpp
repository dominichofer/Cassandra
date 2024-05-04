#include "PossibleMoves.h"
#include <cstdint>

CUDA_CALLABLE Moves PossibleMoves(const Position& pos) noexcept
{
	#ifdef __NVCC__
		return detail::PossibleMoves_x64(pos);
	#elif defined(__AVX512F__)
		return detail::PossibleMoves_AVX512(pos);
	#elif defined(__AVX2__)
		return detail::PossibleMoves_AVX2(pos);
	#else
		return detail::PossibleMoves_x64(pos);
	#endif
}

#ifdef __AVX512F__
Moves detail::PossibleMoves_AVX512(const Position& pos) noexcept
{
	// 6 x SHIFT, 7 x AND, 5 x OR, 1 x NOT
	// = 19 OPs

	// 1 x AND
	const __m512i maskO = _mm512_set1_epi64(pos.Opponent()) & _mm512_set_epULL(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL, 0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
	const __m512i shift1 = _mm512_set_epi64(1, 8, 7, 9, -1, -8, -7, -9);
	const __m512i shift2 = shift1 + shift1;

	// 6 x SHIFT, 5 x AND, 3 x OR
	__m512i flip = maskO & _mm512_rolv_epi64(_mm512_set1_epULL(pos.Player()), shift1);
	__m512i mask = maskO & _mm512_rolv_epi64(maskO, shift1);
	flip |= maskO & _mm512_rolv_epi64(flip, shift1);
	flip |= mask & _mm512_rolv_epi64(flip, shift2);
	flip |= mask & _mm512_rolv_epi64(flip, shift2);
	flip = _mm512_rolv_epi64(flip, shift1);

	// 1 x NOT, 2 x OR, 1 x AND
	return Moves{ ~(P | O) & _mm512_reduce_or_epi64(flip) };
}
#endif

#ifdef __AVX2__
Moves detail::PossibleMoves_AVX2(const Position& pos) noexcept
{
	// 8 x OR, 12 x SHIFT, 11 x AND, 1 x NOT, 4 x Extract
	// = 36 OPs

	// 1 x AND
	const __m256i P = _mm256_set1_epi64x(pos.Player());
	const __m256i O = _mm256_set1_epi64x(pos.Opponent());
	const __m256i maskO = _mm256_and_si256(O, _mm256_set_epi64x(0x7E7E7E7E7E7E7E7EULL, 0xFFFFFFFFFFFFFFFFULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL));
	const __m256i shift1 = _mm256_set_epi64x(1, 8, 7, 9);
	const __m256i shift2 = _mm256_set_epi64x(2, 16, 14, 18);

	// 2 x SHIFT, 2 x AND
	__m256i flip1 = _mm256_and_si256(maskO, _mm256_sllv_epi64(P, shift1)); // flips1 = maskO & (P << shift1)
	__m256i flip2 = _mm256_and_si256(maskO, _mm256_srlv_epi64(P, shift1)); // flips1 = maskO & (P >> shift1)

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 = _mm256_or_si256(flip1, _mm256_and_si256(maskO, _mm256_sllv_epi64(flip1, shift1))); // flip1 |= maskO & (flips1 << shift1)
	flip2 = _mm256_or_si256(flip2, _mm256_and_si256(maskO, _mm256_srlv_epi64(flip2, shift1))); // flip2 |= maskO & (flips2 >> shift1)

	// 2 x SHIFT, 1 x AND
	const __m256i mask1 = _mm256_and_si256(maskO, _mm256_sllv_epi64(maskO, shift1)); // mask1 = maskO & (maskO << shift1);
	const __m256i mask2 = _mm256_srlv_epi64(mask1, shift1); // mask2 = mask1 >> shift1;

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 = _mm256_or_si256(flip1, _mm256_and_si256(mask1, _mm256_sllv_epi64(flip1, shift2))); // flip1 |= mask1 & (flips1 << shift2)
	flip2 = _mm256_or_si256(flip2, _mm256_and_si256(mask2, _mm256_srlv_epi64(flip2, shift2))); // flip2 |= mask2 & (flips2 >> shift2)

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 = _mm256_or_si256(flip1, _mm256_and_si256(mask1, _mm256_sllv_epi64(flip1, shift2))); // flip1 |= mask1 & (flips1 << shift2)
	flip2 = _mm256_or_si256(flip2, _mm256_and_si256(mask2, _mm256_srlv_epi64(flip2, shift2))); // flip2 |= mask2 & (flips2 >> shift2)

	// 2 x SHIFT
	flip1 = _mm256_sllv_epi64(flip1, shift1); // flip1 <<= shift1;
	flip2 = _mm256_srlv_epi64(flip2, shift1); // flip2 >>= shift1;
	
	// 3 x OR, 4 x Extract
	const uint64_t flip = reduce_or(_mm256_or_si256(flip1, flip2)); // flip = reduce_or(flip1 | flip2)

	// 1 x AND, 1 x OR, 1 x NOT
	return Moves{ pos.Empties() & flip };
}
#endif

CUDA_CALLABLE Moves detail::PossibleMoves_x64(const Position& pos) noexcept
{
	// 48 x SHIFT, 38 x AND, 32 x OR, 1 x NOT
	// = 119 OPs
	
	// 1 x AND
	const uint64_t maskO = pos.Opponent() & 0x7E7E7E7E7E7E7E7EULL;

	// 8 x SHIFT, 8 x AND
	uint64_t flip1 = maskO & (pos.Player() << 1);
	uint64_t flip2 = maskO & (pos.Player() >> 1);
	uint64_t flip3 = pos.Opponent() & (pos.Player() << 8);
	uint64_t flip4 = pos.Opponent() & (pos.Player() >> 8);
	uint64_t flip5 = maskO & (pos.Player() << 7);
	uint64_t flip6 = maskO & (pos.Player() >> 7);
	uint64_t flip7 = maskO & (pos.Player() << 9);
	uint64_t flip8 = maskO & (pos.Player() >> 9);

	// 8 x SHIFT, 8 x AND, 8 x OR
	flip1 |= maskO & (flip1 << 1);
	flip2 |= maskO & (flip2 >> 1);
	flip3 |= pos.Opponent() & (flip3 << 8);
	flip4 |= pos.Opponent() & (flip4 >> 8);
	flip5 |= maskO & (flip5 << 7);
	flip6 |= maskO & (flip6 >> 7);
	flip7 |= maskO & (flip7 << 9);
	flip8 |= maskO & (flip8 >> 9);

	// 8 x SHIFT, 4 x AND
	uint64_t mask1 = maskO & (maskO << 1);
	uint64_t mask2 = mask1 >> 1;
	uint64_t mask3 = pos.Opponent() & (pos.Opponent() << 8);
	uint64_t mask4 = mask3 >> 8;
	uint64_t mask5 = maskO & (maskO << 7);
	uint64_t mask6 = mask5 >> 7;
	uint64_t mask7 = maskO & (maskO << 9);
	uint64_t mask8 = mask7 >> 9;

	// 8 x SHIFT, 8 x AND, 8 x OR
	flip1 |= mask1 & (flip1 << 2);
	flip2 |= mask2 & (flip2 >> 2);
	flip3 |= mask3 & (flip3 << 16);
	flip4 |= mask4 & (flip4 >> 16);
	flip5 |= mask5 & (flip5 << 14);
	flip6 |= mask6 & (flip6 >> 14);
	flip7 |= mask7 & (flip7 << 18);
	flip8 |= mask8 & (flip8 >> 18);

	// 8 x SHIFT, 8 x AND, 8 x OR
	flip1 |= mask1 & (flip1 << 2);
	flip2 |= mask2 & (flip2 >> 2);
	flip3 |= mask3 & (flip3 << 16);
	flip4 |= mask4 & (flip4 >> 16);
	flip5 |= mask5 & (flip5 << 14);
	flip6 |= mask6 & (flip6 >> 14);
	flip7 |= mask7 & (flip7 << 18);
	flip8 |= mask8 & (flip8 >> 18);

	// 8 x SHIFT
	flip1 <<= 1;
	flip2 >>= 1;
	flip3 <<= 8;
	flip4 >>= 8;
	flip5 <<= 7;
	flip6 >>= 7;
	flip7 <<= 9;
	flip8 >>= 9;

	return Moves{ pos.Empties() // 1 x AND, 1 x OR, 1 x NOT
		& (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8) }; // 7 x OR
}
