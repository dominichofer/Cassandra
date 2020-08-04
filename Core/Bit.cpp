#pragma once
#include "Bit.h"

uint64_t _mm256_reduce_or_epi64(__m256i x) noexcept
{
	// 1 x PERMUTE (Latency 3)
	// 1 x SHUFFLE, 2 x OR (Latency 1)
	// = 4 OPs (Latency 6)

	// _mm256_permute2x128_si256((a3,a2,a1,a0), (b3,b2,b1,b0), 1) = (a1,a0,a3,a2)
	// _mm256_shuffle_epi32((a7,a6,a5,a4,a3,a2,a1,a0), 0b01'00'11'10) = (a5,a4),(a7,a6),(a1,a0),(a3,a2)
	
	x = _mm256_or_si256(x, _mm256_permute2x128_si256(x, x, 1)); // = (x1|x3, x0|x2), (dito)
	x = _mm256_or_si256(x, _mm256_shuffle_epi32(x, 0b01'00'11'10)); // = (x1|x3|x0|x2), (dito), (dito), (dito)
	return _mm256_cvtsi256_si64(x);

	//__m128i y = _mm_or_si128(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
	//y = _mm_or_si128(y, _mm_unpackhi_epi64(y, y));
	//return _mm_cvtsi128_si64(y);
}

uint32_t _mm256_reduce_add_epi32(__m256i x) noexcept
{
	// 1 x PERMUTE (Latency 3)
	// 2 x SHUFFLE, 3 x ADD (Latency 1)
	// = 6 OPs (Latency 8)

	// _mm256_permute2x128_si256((a7,a6,a5,a4,a3,a2,a1,a0), (b7,b6,b5,b4,b3,b2,b1,b0), 1) = (a3,a2,a1,a0),(a7,a6,a5,a4)
	// _mm256_shuffle_epi32((a7,a6,a5,a4,a3,a2,a1,a0), 0b01'00'11'10) = (a5,a4),(a7,a6),(a1,a0),(a3,a2)
	// _mm256_shuffle_epi32((a7,a6,a5,a4,a3,a2,a1,a0), 0b10'11'00'01) = (a6,a7,a4,a5,a2,a3,a0,a1)

	x = _mm256_add_epi32(x, _mm256_permute2x128_si256(x, x, 1)); // = (x7+x3, x6+x2, x5+x1, x4+x0), (dito)
	x = _mm256_add_epi32(x, _mm256_shuffle_epi32(x, 0b01'00'11'10)); // = (x7+x3+x5+x1, x6+x2+x4+x0), (dito), (dito), (dito)
	x = _mm256_add_epi32(x, _mm256_shuffle_epi32(x, 0b10'11'00'01)); // = (x7+x3+x5+x1+x6+x2+x4+x0, dito, dito, dito, dito, dito, dito, dito)
	return _mm256_cvtsi256_si32(x);
}
