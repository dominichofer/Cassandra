#pragma once
// Predefined macros:
// __GNUC__           Compiler is gcc.
// __clang__          Compiler is clang.
// __INTEL_COMPILER   Compiler is Intel.
// _MSC_VER           Compiler is Visual Studio.
// _M_X64             Microsoft specific macro when targeting 64 bit based machines.
// __x86_64           Defined by GNU C and Sun Studio when targeting 64 bit based machines.

#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(__GNUC__)
	#include <x86intrin.h>
#else
	#error compiler not supported!
#endif

enum class Compiler
{
	gcc, clang, intel, visual_studio
};

#ifdef __GNUC__
constexpr Compiler compiler = Compiler::gcc;
#endif

#ifdef __clang__
constexpr Compiler compiler = Compiler::clang;
#endif

#ifdef __INTEL_COMPILER
constexpr Compiler compiler = Compiler::intel;
#endif

#ifdef _MSC_VER
constexpr Compiler compiler = Compiler::visual_studio;
#endif


#ifndef __AVX2__
	#error This code requires AVX2!
#endif

#if !(defined(_M_X64) || defined(__x86_64))
	#error This code only works on x64!
#endif


#ifdef __AVX__
constexpr bool CPU_has_AVX = true;
#else
constexpr bool CPU_has_AVX = false;
#endif

#ifdef __AVX2__
constexpr bool CPU_has_AVX2 = true;
#else
constexpr bool CPU_has_AVX2 = false;
#endif

#ifdef __AVX512F__
constexpr bool CPU_has_AVX512F = true;
#else
constexpr bool CPU_has_AVX512F = false;
#endif

constexpr bool CPU_has_PopCount = CPU_has_AVX || CPU_has_AVX2;
constexpr bool CPU_has_SSE2 = CPU_has_AVX || CPU_has_AVX2;
constexpr bool CPU_has_SSE4_1 = CPU_has_AVX || CPU_has_AVX2;
constexpr bool CPU_has_SSE4_2 = CPU_has_AVX || CPU_has_AVX2;
