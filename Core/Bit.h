#pragma once
#include "Cuda.h"
#include "Intrin.h"
#include <cstdint>

#ifdef __AVX2__
	inline __m256i not_si256(__m256i b) noexcept
	{
		return _mm256_andnot_si256(b, _mm256_set1_epi64x(-1));
	}

	inline __m256i neg_epi64(__m256i b) noexcept
	{
		return _mm256_sub_epi64(__m256i{0}, b);
	}

	inline uint64_t reduce_or(__m256i b) noexcept
	{
		const __m128i or_128 = _mm_or_si128(_mm256_extracti128_si256(b, 0), _mm256_extracti128_si256(b, 1));
		return _mm_extract_epi64(or_128, 0) | _mm_extract_epi64(or_128, 1);
	}
#endif

// Workaround for nvcc's missing <bit>
#ifndef __NVCC__
	#include <bit>
#else
namespace std
{
	#ifdef __CUDA_ARCH__
		// Number of consecutive 0 bits in the value of x, starting from the most significant bit.
		__device__ inline int countl_zero(uint64_t x) noexcept { return __clzll(x); }

		// Number of consecutive 1 bits in the value of x, starting from the most significant bit.
		__device__ inline int countl_one(uint64_t x) noexcept { return countl_zero(~x); }

		// Number of consecutive 0 bits in the value of x, starting from the least significant bit
		__device__ inline int countr_zero(uint64_t x) noexcept { return __ffsll(x) - 1; }

		// Number of consecutive 1 bits in the value of x, starting from the least significant bit
		__device__ inline int countr_one(uint64_t x) noexcept { return countr_zero(~x); }

		// Number of 1 bits in the value of x.
		__device__ inline int popcount(uint64_t x) noexcept { return __popcll(x); }
	#else
		// Number of consecutive 0 bits in the value of x, starting from the most significant bit.
		__host__ inline int countl_zero(uint64_t x) noexcept { return static_cast<int>(_lzcnt_u64(x)); }

		// Number of consecutive 1 bits in the value of x, starting from the most significant bit.
		__host__ inline int countl_one(uint64_t x) noexcept { return static_cast<int>(countl_zero(~x)); }

		// Number of consecutive 0 bits in the value of x, starting from the least significant bit
		__host__ inline int countr_zero(uint64_t x) noexcept { return static_cast<int>(_tzcnt_u64(x)); }

		// Number of consecutive 1 bits in the value of x, starting from the least significant bit
		__host__ inline int countr_one(uint64_t x) noexcept { return static_cast<int>(countr_zero(~x)); }

		// Number of 1 bits in the value of x.
		__host__ inline int popcount(uint64_t x) noexcept { return static_cast<int>(_mm_popcnt_u64(x)); }
	#endif
}
#endif


CUDA_CALLABLE inline uint64_t GetLSB(uint64_t b) noexcept
{
	#pragma warning(suppress : 4146)
	return b & (0 - b);
}

CUDA_CALLABLE inline void RemoveLSB(uint64_t& b) noexcept
{
	b &= b - 1;
}

CUDA_CALLABLE inline uint64_t BExtr(uint64_t src, unsigned int start, unsigned int len) noexcept
{
	#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
		return _bextr_u64(src, start, len);
	#else
		return (src >> start) & ((1ULL << len) - 1);
	#endif
}

inline uint64_t PDep(uint64_t src, uint64_t mask) noexcept
{
	return _pdep_u64(src, mask);
}

inline uint64_t PExt(uint64_t src, uint64_t mask) noexcept
{
	return _pext_u64(src, mask);
}

CUDA_CALLABLE inline uint64_t BSwap(uint64_t b) noexcept
{
	#if defined(__CUDA_ARCH__)
		return __byte_perm(b, 0, 0x0123);
	#elif defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}
