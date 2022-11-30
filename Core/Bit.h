#pragma once
#include "MoreTypes.h"

#ifndef __NVCC__
	#include <bit>
#else
namespace std
{
	#ifdef __CUDA_ARCH__
		// Number of consecutive 0 bits in the value of x, starting from the most significant bit.
		__device__ inline int countl_zero(uint64 x) noexcept { return __clzll(x); }

		// Number of consecutive 1 bits in the value of x, starting from the most significant bit.
		__device__ inline int countl_one(uint64 x) noexcept { return countl_zero(~x); }

		// Number of consecutive 0 bits in the value of x, starting from the least significant bit
		__device__ inline int countr_zero(uint64 x) noexcept { return __ffsll(x) - 1; }

		// Number of consecutive 1 bits in the value of x, starting from the least significant bit
		__device__ inline int countr_one(uint64 x) noexcept { return countr_zero(~x); }

		// Number of 1 bits in the value of x.
		__device__ inline int popcount(uint64 x) noexcept { return __popcll(x); }
	#else
		// Number of consecutive 0 bits in the value of x, starting from the most significant bit.
		__host__ inline int countl_zero(uint64 x) noexcept { return static_cast<int>(_lzcnt_u64(x)); }

		// Number of consecutive 1 bits in the value of x, starting from the most significant bit.
		__host__ inline int countl_one(uint64 x) noexcept { return static_cast<int>(countl_zero(~x)); }

		// Number of consecutive 0 bits in the value of x, starting from the least significant bit
		__host__ inline int countr_zero(uint64 x) noexcept { return static_cast<int>(_tzcnt_u64(x)); }

		// Number of consecutive 1 bits in the value of x, starting from the least significant bit
		__host__ inline int countr_one(uint64 x) noexcept { return static_cast<int>(countr_zero(~x)); }

		// Number of 1 bits in the value of x.
		__host__ inline int popcount(uint64 x) noexcept { return static_cast<int>(_mm_popcnt_u64(x)); }
	#endif
}
#endif

#ifdef __CUDA_ARCH__
	#define CUDA_CALLABLE __host__ __device__
#else
	#define CUDA_CALLABLE
#endif


CUDA_CALLABLE inline uint64 GetLSB(uint64 b) noexcept
{
	#pragma warning(suppress : 4146)
	return b & -b;
}

CUDA_CALLABLE inline void RemoveLSB(uint64& b) noexcept
{
	b &= b - 1;
}

CUDA_CALLABLE inline uint64 BExtr(uint64 src, uint start, uint len) noexcept
{
	#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
		return _bextr_u64(src, start, len);
	#else
		return (src >> start) & ((1ULL << len) - 1);
	#endif
}

inline uint64 PDep(uint64 src, uint64 mask) noexcept
{
	return _pdep_u64(src, mask);
}

inline uint64 PExt(uint64 src, uint64 mask) noexcept
{
	return _pext_u64(src, mask);
}

CUDA_CALLABLE inline uint64 BSwap(uint64 b) noexcept
{
	#if defined(__CUDA_ARCH__)
		return __byte_perm(b, 0, 0x0123);
	#elif defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}
