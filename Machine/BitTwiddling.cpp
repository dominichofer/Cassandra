#pragma once
#include "Machine/BitTwiddling.h"
#include "MacrosHell.h"

[[nodiscard]]
unsigned int BitScanLSB(const uint64_t mask) noexcept
{
	// BitScanLSB(0) may be undefined.

	#if defined(_MSC_VER)
		unsigned long index;
		_BitScanForward64(&index, mask);
		return index;
	#elif defined(__GNUC__)
		return __builtin_ctzll(mask); // __builtin_ctzll(0) is undefined
	#endif
}

[[nodiscard]]
std::size_t CountTrailingZeros(const uint64_t mask) noexcept
{
	// CountTrailingZeros(0) may be undefined.

	#if defined(_MSC_VER)
		return _tzcnt_u64(mask); // _tzcnt_u64(0) is undefined
	#elif defined(__GNUC__)
		return __builtin_ctzll(mask); // __builtin_ctzll(0) is undefined
	#endif
}

[[nodiscard]]
uint64_t GetLSB(const uint64_t b) noexcept
{
	#ifdef HAS_BLSI
		return _blsi_u64(b);
	#else
	#pragma warning(suppress : 4146)
		return b & -b;
	#endif
}

void RemoveLSB(uint64_t& b) noexcept
{
	#ifdef HAS_BLSR
		detail::RemoveLSB_intrinsic(b);
	#else
		detail::RemoveLSB_generic(b);
	#endif
}

[[nodiscard]]
std::size_t PopCount(uint64_t b) noexcept
{
	#ifdef HAS_POPCNT
		return detail::PopCount_intrinsic(b);
	#else
		return detail::PopCount_generic(b);
	#endif
}

[[nodiscard]]
uint64_t PDep(uint64_t src, uint64_t mask) noexcept
{
	#ifdef HAS_PDEP
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
	#ifdef HAS_PEXT
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
	// 1 x PERMUTE, 1 x SHUFFLE, 2 x OR
	// = 4 OPs
	__m256i x0 = _mm256_permute2x128_si256(x, x, 1);
	__m256i x1 = _mm256_or_si256(x, x0);
	__m256i x2 = _mm256_shuffle_epi32(x1, 0b01001110);
	__m256i x3 = _mm256_or_si256(x1, x2);
	return _mm_cvtsi128_si64(_mm256_castsi256_si128(x3));
}