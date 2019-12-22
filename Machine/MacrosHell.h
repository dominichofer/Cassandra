#pragma once
#include <cassert>
#include <cstdint>

// Predefined macros:
// __GNUC__           Compiler is gcc, clang or Intel on Linux.
// __INTEL_COMPILER   Compiler is Intel.
// _MSC_VER           Compiler is MSVC or Intel on ExclusiveIntervals.
// _WIN32             Building on ExclusiveIntervals (any).
// _WIN64             Building on ExclusiveIntervals 64 bit.
// _M_X64             Microsoft specific macro for 64 bit based machines.
// __x86_64           Defined by GNU C and Sun Studio for 64 bit based machines.

#if !(defined(_WIN64) || defined(__x86_64))
	#error This code only works on x64!
#endif

#if defined(_MSC_VER)
	#include <intrin.h>
	#pragma intrinsic(_BitScanForward64)
	#pragma intrinsic(_BitScanReverse64)
#elif defined(__GNUC__)
	#include <x86intrin.h>
#else
	#error compiler not supported
#endif

// #####################################
// ##    CPU specific optimizations   ##
// #####################################
//#define __core2__
//#define __corei7__
//#define __corei7_avx__
//#define __core_avx_i__
//#define __core_avx2__
//#define __bdver1__

//#define HAS_MMX
//#define HAS_SSE
//#define HAS_SSE2
//#define HAS_SSE3
//#define HAS_SSSE3
//#define HAS_SSE4_1
//#define HAS_SSE4_2
//#define HAS_SSE4a
//#define HAS_AVX
//#define HAS_AVX2
//#define HAS_BMI1
//#define HAS_BMI2
//#define HAS_ABM
//#define HAS_TBM
//#define HAS_POPCNT
//#define HAS_LZCNT
// #####################################

// CPUs instruction sets
#ifdef __core2__
#define HAS_MMX
#define HAS_SSE
#define HAS_SSE2
#define HAS_SSE3
#define HAS_SSSE3
#define HAS_SSE4_1
#endif
#ifdef __corei7__
#define HAS_MMX
#define HAS_SSE
#define HAS_SSE2
#define HAS_SSE3
#define HAS_SSSE3
#define HAS_SSE4_1
#define HAS_SSE4_2
#endif
#if defined(__corei7_avx__) || defined(__core_avx_i__)
#define HAS_MMX
#define HAS_SSE
#define HAS_SSE2
#define HAS_SSE3
#define HAS_SSSE3
#define HAS_SSE4_1
#define HAS_SSE4_2
#endif
#ifdef __core_avx2__
#define HAS_MMX
#define HAS_SSE
#define HAS_SSE2
#define HAS_SSE3
#define HAS_SSSE3
#define HAS_SSE4_1
#define HAS_SSE4_2
#define HAS_AVX
#define HAS_AVX2
#define HAS_BMI1
#define HAS_BMI2
#define HAS_POPCNT
#define HAS_LZCNT
#endif
#ifdef __bdver1__
#define HAS_MMX
#define HAS_SSE
#define HAS_SSE2
#define HAS_SSE3
#define HAS_SSSE3
#define HAS_SSE4_1
#define HAS_SSE4_2
#define HAS_SSE4a
#define HAS_AVX
#define HAS_BMI1
#define HAS_ABM
#define HAS_TBM
#endif

// CPU instruction implications
#ifdef HAS_BMI1
#define HAS_BEXTR	// Bit Field Extract
#define HAS_BLSI	// Extract Lowest Set Isolated Bit  (x & -x)
#define HAS_BLSMASK	// Get mask up to lowest set bit    (x ^ (x - 1))
#define HAS_BLSR	// Reset lowest set bit             (x & (x - 1))
#define HAS_LZCNT	// Leading Zero Count
#define HAS_TZCNT	// Trailing Zero Count
#endif
#ifdef HAS_BMI2
#define HAS_BZHI // Zero high bits starting with specified bit position
#define HAS_PDEP // Parallel bits deposit
#define HAS_PEXT // Parallel bits extract
#endif
#ifdef HAS_ABM
#define HAS_POPCNT // Population count
#define HAS_LZCNT  // Leading Zero Count
#endif
#ifdef HAS_TBM
#define HAS_BEXTR	// Bit Field Extract
#define HAS_BLCFILL	// Fill from lowest clear bit               ( x &  (x + 1))
#define HAS_BLCI	// Isolate lowest clear bit                 ( x | ~(x + 1))
#define HAS_BLCIC	// Isolate lowest clear bit and complement  (~x &  (x + 1))
#define HAS_BLCMASK	// Mask from lowest clear bit               ( x ^  (x + 1))
#define HAS_BLCS	// Set lowest clear bit                     ( x |  (x + 1))
#define HAS_BLSFILL	// Fill from lowest set bit                 ( x |  (x - 1))
#define HAS_BLSIC	// Isolate lowest set bit and complement    (~x |  (x - 1))
#define HAS_T1MSKC	// Inverse mask from trailing ones          (~x |  (x + 1))
#define HAS_TZMSK	// Mask from trailing zeros                 (~x &  (x - 1))
#endif
