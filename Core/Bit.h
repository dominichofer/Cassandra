#pragma once
#include <cstdint>

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

#ifndef __AVX2__
	#error This code requires AVX2!
#endif

#if !(defined(_M_X64) || defined(__x86_64))
	#error This code requires x64!
#endif

// TODO: Rename!
//[[nodiscard]]
//constexpr uint64_t Horizontal(int i) noexcept { return 0xFFULL << (i * 8); }
//
//[[nodiscard]]
//constexpr uint64_t Vertical(int i) noexcept { return 0x0101010101010101ULL << i; }
//
//[[nodiscard]]
//constexpr uint64_t Diagonal(int i) noexcept
//{
//	if (i > 0)
//		return 0x8040201008040201ULL << (8 * i);
//	return 0x8040201008040201ULL >> (8 * i);
//}

[[nodiscard]]
inline int countr_zero(const uint64_t mask) noexcept
{
	#if defined(_MSC_VER)
		return _tzcnt_u64(mask);
	#elif defined(__GNUC__)
		return __builtin_ia32_tzcnt_u64(mask);
	#endif
}

[[nodiscard]]
inline uint64_t GetLSB(const uint64_t b) noexcept
{
	#pragma warning(suppress : 4146)
	return b & -b;
}

inline void RemoveLSB(uint64_t& b) noexcept
{
	b &= b - 1;
}

[[nodiscard]]
inline int popcount(uint64_t b) noexcept
{
	return __builtin_popcountll(b);
}

[[nodiscard]]
inline uint64_t BExtr(const uint64_t src, const unsigned int start, unsigned int len) noexcept
{
	#ifdef _MSC_VER
		return _bextr_u64(src, start, len);
	#else
		return (src >> start) & ((1ULL << len) - 1);
	#endif
}

[[nodiscard]]
inline uint64_t PDep(uint64_t src, uint64_t mask) noexcept
{
	return _pdep_u64(src, mask);
}

[[nodiscard]]
inline uint64_t PExt(uint64_t src, uint64_t mask) noexcept
{
	return _pext_u64(src, mask);
}

[[nodiscard]]
inline uint64_t BSwap(const uint64_t b) noexcept
{
	#if defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}

#ifdef __GNUC__ 
	#ifdef __clang__
		[[nodiscard]] inline uint64_t _mm256_cvtsi256_si64(__m256i a) { return _mm_cvtsi128_si64(_mm256_castsi256_si128(a)); }
	#else
		[[nodiscard]] inline uint32_t _mm256_cvtsi256_si32(__m256i a) { return _mm_cvtsi128_si32(_mm256_castsi256_si128(a)); }
	#endif
#endif

#ifdef _MSC_VER
    [[nodiscard]] inline __m128i operator~(const __m128i& a) noexcept { return _mm_andnot_si128(a, _mm_set1_epi64x(-1)); }

    [[nodiscard]] inline __m128i operator+(const __m128i& a, const __m128i& b) noexcept { return _mm_add_epi64(a, b); }
    [[nodiscard]] inline __m128i operator-(const __m128i& a, const __m128i& b) noexcept { return _mm_sub_epi64(a, b); }
    [[nodiscard]] inline __m128i operator&(const __m128i& a, const __m128i& b) noexcept { return _mm_and_si128(a, b); }
    [[nodiscard]] inline __m128i operator|(const __m128i& a, const __m128i& b) noexcept { return _mm_or_si128(a, b); }
    [[nodiscard]] inline __m128i operator^(const __m128i& a, const __m128i& b) noexcept { return _mm_xor_si128(a, b); }
    [[nodiscard]] inline __m128i operator<<(const __m128i& a, const int b) noexcept { return _mm_slli_epi64(a, b); }
    [[nodiscard]] inline __m128i operator>>(const __m128i& a, const int b) noexcept { return _mm_srli_epi64(a, b); }
	[[nodiscard]] inline __m128i operator<<(const __m128i& a, const __m128i& b) noexcept { return _mm_sllv_epi64(a, b); }
	[[nodiscard]] inline __m128i operator>>(const __m128i& a, const __m128i& b) noexcept { return _mm_srlv_epi64(a, b); }

    inline __m128i& operator+=(__m128i& a, const __m128i& b) noexcept { return a = a + b; }
    inline __m128i& operator-=(__m128i& a, const __m128i& b) noexcept { return a = a - b; }
    inline __m128i& operator&=(__m128i& a, const __m128i& b) noexcept { return a = a & b; }
    inline __m128i& operator|=(__m128i& a, const __m128i& b) noexcept { return a = a | b; }
    inline __m128i& operator^=(__m128i& a, const __m128i& b) noexcept { return a = a ^ b; }
    inline __m128i& operator<<=(__m128i& a, const int b) noexcept { return a = a << b; }
    inline __m128i& operator>>=(__m128i& a, const int b) noexcept { return a = a >> b; }
	inline __m128i& operator<<=(__m128i& a, const __m128i& b) noexcept { return a = a << b; }
	inline __m128i& operator>>=(__m128i& a, const __m128i& b) noexcept { return a = a >> b; }
	[[nodiscard]] inline __m128i operator-(const __m128i& a) noexcept { return _mm_setzero_si128() - a; }

	[[nodiscard]] inline __m128i operator==(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpeq_epi64(a, b); }
	[[nodiscard]] inline __m128i operator!=(const __m128i& a, const __m128i& b) noexcept { return ~(a == b); }
    [[nodiscard]] inline __m128i operator>(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpgt_epi64(a, b); }
    [[nodiscard]] inline __m128i operator<(const __m128i& a, const __m128i& b) noexcept { return b > a; }
    [[nodiscard]] inline __m128i operator>=(const __m128i& a, const __m128i& b) noexcept { return ~(a < b); }
    [[nodiscard]] inline __m128i operator<=(const __m128i& a, const __m128i& b) noexcept { return ~(a > b); }


	[[nodiscard]] inline __m256i operator~(const __m256i& a) noexcept { return _mm256_andnot_si256(a, _mm256_set1_epi64x(-1)); }

    [[nodiscard]] inline __m256i operator+(const __m256i& a, const __m256i& b) noexcept { return _mm256_add_epi64(a, b); }
    [[nodiscard]] inline __m256i operator-(const __m256i& a, const __m256i& b) noexcept { return _mm256_sub_epi64(a, b); }
    [[nodiscard]] inline __m256i operator&(const __m256i& a, const __m256i& b) noexcept { return _mm256_and_si256(a, b); }
    [[nodiscard]] inline __m256i operator|(const __m256i& a, const __m256i& b) noexcept { return _mm256_or_si256(a, b); }
    [[nodiscard]] inline __m256i operator^(const __m256i& a, const __m256i& b) noexcept { return _mm256_xor_si256(a, b); }
    [[nodiscard]] inline __m256i operator<<(const __m256i& a, const int b) noexcept { return _mm256_slli_epi64(a, b); }
    [[nodiscard]] inline __m256i operator>>(const __m256i& a, const int b) noexcept { return _mm256_srli_epi64(a, b); }
	[[nodiscard]] inline __m256i operator<<(const __m256i& a, const __m256i& b) noexcept { return _mm256_sllv_epi64(a, b); }
	[[nodiscard]] inline __m256i operator>>(const __m256i& a, const __m256i& b) noexcept { return _mm256_srlv_epi64(a, b); }

    inline __m256i& operator+=(__m256i& a, const __m256i& b) noexcept { return a = a + b; }
    inline __m256i& operator-=(__m256i& a, const __m256i& b) noexcept { return a = a - b; }
    inline __m256i& operator&=(__m256i& a, const __m256i& b) noexcept { return a = a & b; }
    inline __m256i& operator|=(__m256i& a, const __m256i& b) noexcept { return a = a | b; }
    inline __m256i& operator^=(__m256i& a, const __m256i& b) noexcept { return a = a ^ b; }
    inline __m256i& operator<<=(__m256i& a, const int b) noexcept { return a = a << b; }
    inline __m256i& operator>>=(__m256i& a, const int b) noexcept { return a = a >> b; }
	inline __m256i& operator<<=(__m256i& a, const __m256i& b) noexcept { return a = a << b; }
	inline __m256i& operator>>=(__m256i& a, const __m256i& b) noexcept { return a = a >> b; }
	[[nodiscard]] inline __m256i operator-(const __m256i& a) noexcept { return _mm256_setzero_si256() - a; }

	[[nodiscard]] inline __m256i operator==(const __m256i& a, const __m256i& b) noexcept { return _mm256_cmpeq_epi64(a, b); }
	[[nodiscard]] inline __m256i operator!=(const __m256i& a, const __m256i& b) noexcept { return ~(a == b); }
	[[nodiscard]] inline __m256i operator>(const __m256i& a, const __m256i& b) noexcept { return _mm256_cmpgt_epi64(a, b); }
	[[nodiscard]] inline __m256i operator<(const __m256i& a, const __m256i& b) noexcept { return b > a; }
	[[nodiscard]] inline __m256i operator>=(const __m256i& a, const __m256i& b) noexcept { return ~(a < b); }
	[[nodiscard]] inline __m256i operator<=(const __m256i& a, const __m256i& b) noexcept { return ~(a > b); }

	uint64_t _mm256_reduce_or_epi64(__m256i) noexcept;
	uint32_t _mm256_reduce_add_epi32(__m256i) noexcept;

	#ifdef __AVX512F__
		[[nodiscard]] inline __m512i operator~(const __m512i& a) noexcept { return _mm512_xor_si512(a, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFULL)); }

		[[nodiscard]] inline __m512i operator+(const __m512i& a, const __m512i& b) noexcept { return _mm512_add_epi64(a, b); }
		[[nodiscard]] inline __m512i operator-(const __m512i& a, const __m512i& b) noexcept { return _mm512_sub_epi64(a, b); }
		[[nodiscard]] inline __m512i operator&(const __m512i& a, const __m512i& b) noexcept { return _mm512_and_si512(a, b); }
		[[nodiscard]] inline __m512i operator|(const __m512i& a, const __m512i& b) noexcept { return _mm512_or_si512(a, b); }
		[[nodiscard]] inline __m512i operator^(const __m512i& a, const __m512i& b) noexcept { return _mm512_xor_si512(a, b); }
		[[nodiscard]] inline __m512i operator<<(const __m512i& a, const int b) noexcept { return _mm512_slli_epi64(a, b); }
		[[nodiscard]] inline __m512i operator>>(const __m512i& a, const int b) noexcept { return _mm512_srli_epi64(a, b); }

		inline __m512i operator+=(__m512i& a, const __m512i& b) noexcept { return a = a + b; }
		inline __m512i operator-=(__m512i& a, const __m512i& b) noexcept { return a = a - b; }
		inline __m512i operator&=(__m512i& a, const __m512i& b) noexcept { return a = a & b; }
		inline __m512i operator|=(__m512i& a, const __m512i& b) noexcept { return a = a | b; }
		inline __m512i operator^=(__m512i& a, const __m512i& b) noexcept { return a = a ^ b; }
		inline __m512i operator<<=(__m512i& a, const int b) noexcept { return a = a << b; }
		inline __m512i operator>>=(__m512i& a, const int b) noexcept { return a = a >> b; }
	#endif
#endif
