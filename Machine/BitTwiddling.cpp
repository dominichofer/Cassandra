#pragma once
#include "Machine/BitTwiddling.h"
#include "MacrosHell.h"

[[nodiscard]]
uint64_t GetLSB(const uint64_t b) noexcept
{
	#ifdef __AVX2__
		return _blsi_u64(b);
	#else
		#pragma warning(suppress : 4146)
		return b & -b;
	#endif
}

[[nodiscard]]
std::size_t CountTrailingZeros(const uint64_t mask) noexcept
{
	// CountTrailingZeros(0) == 64

	#if defined(_MSC_VER)
		return _tzcnt_u64(mask); // _tzcnt_u64(0) == 64
	#elif defined(__GNUC__)
		return __builtin_ia32_tzcnt_u64(mask); // __builtin_ia32_tzcnt_u64(0) == 64
	#endif
}

[[nodiscard]]
std::size_t PopCount(uint64_t b) noexcept
{
	if constexpr (CPU_has_PopCount)
		return detail::PopCount_intrinsic(b);
	return detail::PopCount_generic(b);
}

[[nodiscard]]
uint64_t PDep(uint64_t src, uint64_t mask) noexcept
{
	#ifdef __AVX2__
		return _pdep_u64(src, mask);
	#else
		uint64_t res = 0;
		for (uint64_t bb = 1; mask != 0u; bb += bb)
		{
			if ((src & bb) != 0u)
				res |= GetLSB(mask);
			RemoveLSB(mask);
		}
		return res;
	#endif
}

[[nodiscard]]
uint64_t PExt(uint64_t src, uint64_t mask) noexcept
{
	#ifdef __AVX2__
		return _pext_u64(src, mask);
	#else
		uint64_t res = 0;
		for (uint64_t bb = 1; mask != 0u; bb += bb)
		{
			if ((src & GetLSB(mask)) != 0u)
				res |= bb;
			RemoveLSB(mask);
		}
		return res;
	#endif
}

[[nodiscard]]
uint64_t BSwap(const uint64_t b) noexcept
{
	#if defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}

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
