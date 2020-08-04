#include "PossibleMoves.h"
#include "Bit.h"

Moves PossibleMoves(const Position& pos) noexcept
{
	#if defined(__AVX512F__)
		return detail::PossibleMoves_AVX512(pos);
	#elif defined(__AVX2__)
		return detail::PossibleMoves_AVX2(pos);
	#elif defined(__SSE2__)
		return detail::PossibleMoves_SSE2(pos);
	#else
		return detail::PossibleMoves_x64(pos);
	#endif
}

#if defined(__AVX512F__)
Moves detail::PossibleMoves_AVX512(const Position& pos) noexcept
{
	// 6 x SHIFT, 7 x AND, 5 x OR, 1 x NOT
	// = 19 OPs

	// 1 x AND
	const __m512i maskO = _mm512_set1_epi64(pos.O) & _mm512_set_epi64(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL, 0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
	const __m512i shift1 = _mm512_set_epi64(1, 8, 7, 9, -1, -8, -7, -9);
	const __m512i shift2 = shift1 + shift1;

	// 6 x SHIFT, 5 x AND, 3 x OR
	__m512i flip = maskO & _mm512_rolv_epi64(_mm512_set1_epi64(pos.P), shift1);
	__m512i mask = maskO & _mm512_rolv_epi64(maskO, shift1);
	flip |= maskO & _mm512_rolv_epi64(flip, shift1);
	flip |= mask & _mm512_rolv_epi64(flip, shift2);
	flip |= mask & _mm512_rolv_epi64(flip, shift2);
	flip = _mm512_rolv_epi64(flip, shift1);

	// 1 x NOT, 2 x OR, 1 x AND
	return ~(P | O) & _mm512_reduce_or_epi64(flip);
}
#endif

#if defined(__AVX2__)
Moves detail::PossibleMoves_AVX2(const Position& pos) noexcept
{
	// 1 x PERMUTE, 1 x SHUFFLE, 10 x OR, 12 x SHIFT, 11 x AND, 1 x NOT
	// = 36 OPs

	// 1 x AND
	const __m256i maskO = _mm256_set1_epi64x(pos.O) & _mm256_set_epi64x(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
	const __m256i shift = _mm256_set_epi64x(1, 8, 7, 9);
	const __m256i PP = _mm256_set1_epi64x(pos.P);

	// 2 x SHIFT, 2 x AND
	__m256i flip1 = maskO & _mm256_sllv_epi64(PP, shift);
	__m256i flip2 = maskO & _mm256_srlv_epi64(PP, shift);

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 |= maskO & _mm256_sllv_epi64(flip1, shift);
	flip2 |= maskO & _mm256_srlv_epi64(flip2, shift);

	// 2 x SHIFT, 1 x AND
	__m256i mask1 = maskO & _mm256_sllv_epi64(maskO, shift);
	__m256i mask2 = _mm256_srlv_epi64(mask1, shift);

	const __m256i shift2 = shift + shift;
	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 |= mask1 & _mm256_sllv_epi64(flip1, shift2);
	flip2 |= mask2 & _mm256_srlv_epi64(flip2, shift2);

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 |= mask1 & _mm256_sllv_epi64(flip1, shift2);
	flip2 |= mask2 & _mm256_srlv_epi64(flip2, shift2);

	// 2 x SHIFT
	flip1 = _mm256_sllv_epi64(flip1, shift);
	flip2 = _mm256_srlv_epi64(flip2, shift);
	
	// 1 x NOT, 2 x OR, 1 x AND, 1 x function call
	return pos.Empties() & _mm256_reduce_or_epi64(flip1 | flip2);
}
#endif

#if defined(__AVX2__)
Moves detail::PossibleMoves_SSE2(const Position& pos) noexcept
{
	// 30 x SHIFT, 28 x AND, 21 x OR, 1 x NOT, 2 x BSWAP
	// = 82 OPs

	// 2 x MOV, 2 x BSWAP
	const __m128i PP = _mm_set_epi64x(BSwap(pos.P), pos.P);
	const __m128i OO = _mm_set_epi64x(BSwap(pos.O), pos.O);

	// 2 x AND
	const uint64_t maskO = pos.O & 0x7E7E7E7E7E7E7E7EULL;
	const __m128i  maskOO = OO & _mm_set1_epi64x(0x7E7E7E7E7E7E7E7EULL);

	// 5 x SHIFT, 5 x AND
	uint64_t flip1 = maskO & (pos.P << 1);
	uint64_t flip2 = maskO & (pos.P >> 1);
	__m128i flip3 = OO & _mm_slli_epi64(PP, 8);
	__m128i flip4 = maskOO & _mm_slli_epi64(PP, 7);
	__m128i flip5 = maskOO & _mm_srli_epi64(PP, 7);

	// 5 x SHIFT, 5 x AND, 5 x OR
	flip1 |= maskO & (flip1 << 1);
	flip2 |= maskO & (flip2 >> 1);
	flip3 |= OO & _mm_slli_epi64(flip3, 8);
	flip4 |= maskOO & _mm_slli_epi64(flip4, 7);
	flip5 |= maskOO & _mm_srli_epi64(flip5, 7);

	// 5 x SHIFT, 5 x AND
	uint64_t mask1 = maskO & (maskO << 1);
	uint64_t mask2 = (mask1 >> 1);
	__m128i mask3 = OO & _mm_slli_epi64(OO, 8);
	__m128i mask4 = maskOO & _mm_slli_epi64(maskOO, 7);
	__m128i mask5 = _mm_srli_epi64(mask4, 7);

	// 5 x SHIFT, 5 x AND, 5 x OR
	flip1 |= mask1 & (flip1 << 2);
	flip2 |= mask2 & (flip2 >> 2);
	flip3 |= mask3 & _mm_slli_epi64(flip3, 16);
	flip4 |= mask4 & _mm_slli_epi64(flip4, 14);
	flip5 |= mask5 & _mm_srli_epi64(flip5, 14);

	// 5 x SHIFT, 5 x AND, 5 x OR
	flip1 |= mask1 & (flip1 << 2);
	flip2 |= mask2 & (flip2 >> 2);
	flip3 |= mask3 & _mm_slli_epi64(flip3, 16);
	flip4 |= mask4 & _mm_slli_epi64(flip4, 14);
	flip5 |= mask5 & _mm_srli_epi64(flip5, 14);

	// 5 x SHIFT
	flip1 <<= 1;
	flip2 >>= 1;
	flip3 = _mm_slli_epi64(flip3, 8);
	flip4 = _mm_slli_epi64(flip4, 7);
	flip5 = _mm_srli_epi64(flip5, 7);

	// 2 x OR
	flip3 |= flip4 | flip5;

	// 1 x AND, 4 x OR, 1 x NOT
	return pos.Empties() & (flip1 | flip2 | BSwap(_mm_extract_epi64(flip3, 1)) | _mm_extract_epi64(flip3, 0));
}
#endif

Moves detail::PossibleMoves_x64(const Position& pos) noexcept
{
	// 48 x SHIFT, 42 x AND, 32 x OR, 1 x NOT
	// = 123 OPs
	
	// 1 x AND
	const uint64_t maskO = pos.O & 0x7E7E7E7E7E7E7E7EULL;

	// 8 x SHIFT, 8 x AND
	uint64_t flip1 = maskO & (pos.P << 1);
	uint64_t flip2 = maskO & (pos.P >> 1);
	uint64_t flip3 = pos.O & (pos.P << 8);
	uint64_t flip4 = pos.O & (pos.P >> 8);
	uint64_t flip5 = maskO & (pos.P << 7);
	uint64_t flip6 = maskO & (pos.P >> 7);
	uint64_t flip7 = maskO & (pos.P << 9);
	uint64_t flip8 = maskO & (pos.P >> 9);

	// 8 x SHIFT, 8 x AND, 8 x OR
	flip1 |= maskO & (flip1 << 1);
	flip2 |= maskO & (flip2 >> 1);
	flip3 |= pos.O & (flip3 << 8);
	flip4 |= pos.O & (flip4 >> 8);
	flip5 |= maskO & (flip5 << 7);
	flip6 |= maskO & (flip6 >> 7);
	flip7 |= maskO & (flip7 << 9);
	flip8 |= maskO & (flip8 >> 9);

	// 8 x SHIFT, 8 x AND
	uint64_t mask1 = maskO & (maskO << 1);
	uint64_t mask2 = mask1 >> 1;
	uint64_t mask3 = pos.O & (pos.O << 8);
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

	return pos.Empties() // 1 x AND, 1 x OR, 1 x NOT
		& (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8); // 7 x OR
}