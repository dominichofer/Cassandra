#pragma once
#include "MacrosHell.h"
#include <cassert>
#include <cstdint>
#include <cstddef>

// TODO: Rename!
[[nodiscard]]
constexpr uint64_t Horizontal(int i) noexcept { return 0xFFULL << i; }

[[nodiscard]]
constexpr uint64_t Vertical(int i) noexcept { return 0x0101010101010101ULL << i; }

[[nodiscard]]
constexpr uint64_t Diagonal(int i) noexcept
{
	if (i > 0)
		return 0x8040201008040201ULL << (8 * i);
	return 0x8040201008040201ULL >> (8 * i);
}


[[nodiscard]]
inline std::size_t CountLeadingZeros(const uint64_t mask) noexcept
{
	// CountLeadingZeros(0) == 64

	#if defined(_MSC_VER)
		return _lzcnt_u64(mask); // _lzcnt_u64(0) == 64
	#elif defined(__GNUC__)
		return __builtin_ia32_lzcnt_u64(mask); // __builtin_ia32_lzcnt_u64(0) == 64
	#endif
}

[[nodiscard]]
std::size_t CountTrailingZeros(const uint64_t mask) noexcept;

[[nodiscard]]
inline std::size_t BitScanLSB(const uint64_t mask) noexcept
{
	return CountTrailingZeros(mask);
}

[[nodiscard]]
inline std::size_t BitScanMSB(const uint64_t mask) noexcept
{
	return CountLeadingZeros(mask) ^ 63ULL;
}

[[nodiscard]]
uint64_t GetLSB(uint64_t) noexcept;

[[nodiscard]]
inline uint64_t GetMSB(const uint64_t b) noexcept
{
	if (b == 0u) // TODO: Can we get rid of this?
		return 0;
	return 0x8000000000000000ULL >> CountLeadingZeros(b);
}

inline void RemoveLSB(uint64_t& b) noexcept { b &= b - 1; }

inline void RemoveMSB(uint64_t& b) noexcept { b ^= GetMSB(b); }

namespace detail
{
#ifdef __AVX2__
	[[nodiscard]]
	inline std::size_t PopCount_intrinsic(uint64_t b) noexcept
	{
		return __builtin_popcountll(b);
	}
#else
	std::size_t PopCount_intrinsic(uint64_t b) noexcept;
#endif

	[[nodiscard]]
	inline std::size_t PopCount_generic(uint64_t b) noexcept
	{
		b -= (b >> 1) & 0x5555555555555555ULL;
		b = ((b >> 2) & 0x3333333333333333ULL) + (b & 0x3333333333333333ULL);
		b = ((b >> 4) + b) & 0x0F0F0F0F0F0F0F0FULL;
		return (b * 0x0101010101010101ULL) >> 56;
	}
}

[[nodiscard]]
std::size_t PopCount(uint64_t b) noexcept;

[[nodiscard]]
inline uint64_t BExtr(const uint64_t src, const unsigned int start, unsigned int len) noexcept
{
	#ifdef __AVX2__
		return _bextr_u64(src, start, len);
	#else
		return (src >> start) & ((1ULL << len) - 1);
	#endif
}

[[nodiscard]]
inline uint64_t BZHI(const uint64_t src, const uint32_t index) noexcept
{
	#ifdef __AVX2__
		return _bzhi_u64(src, index);
	#else
		return src & ((1ULL << index) - 1);
	#endif
}

[[nodiscard]]
uint64_t PDep(uint64_t src, uint64_t mask) noexcept;

[[nodiscard]]
uint64_t PExt(uint64_t src, uint64_t mask) noexcept;

[[nodiscard]]
uint64_t BSwap(const uint64_t b) noexcept;

#if defined(_MSC_VER)
    #ifdef __AVX2__
        [[nodiscard]] inline __m128i operator~(const __m128i& a) noexcept { return _mm_andnot_si128(a, _mm_set1_epi64x(0xFFFFFFFFFFFFFFFFULL)); }

        [[nodiscard]] inline __m128i operator+(const __m128i& a, const __m128i& b) noexcept { return _mm_add_epi64(a, b); }
        [[nodiscard]] inline __m128i operator-(const __m128i& a, const __m128i& b) noexcept { return _mm_sub_epi64(a, b); }
        [[nodiscard]] inline __m128i operator&(const __m128i& a, const __m128i& b) noexcept { return _mm_and_si128(a, b); }
        [[nodiscard]] inline __m128i operator|(const __m128i& a, const __m128i& b) noexcept { return _mm_or_si128(a, b); }
        [[nodiscard]] inline __m128i operator^(const __m128i& a, const __m128i& b) noexcept { return _mm_xor_si128(a, b); }
        [[nodiscard]] inline __m128i operator<<(const __m128i& a, const int b) noexcept { return _mm_slli_epi64(a, b); }
        [[nodiscard]] inline __m128i operator>>(const __m128i& a, const int b) noexcept { return _mm_srli_epi64(a, b); }

        inline __m128i& operator+=(__m128i& a, const __m128i& b) noexcept { return a = a + b; }
        inline __m128i& operator-=(__m128i& a, const __m128i& b) noexcept { return a = a - b; }
        inline __m128i& operator&=(__m128i& a, const __m128i& b) noexcept { return a = a & b; }
        inline __m128i& operator|=(__m128i& a, const __m128i& b) noexcept { return a = a | b; }
        inline __m128i& operator^=(__m128i& a, const __m128i& b) noexcept { return a = a ^ b; }
        inline __m128i& operator<<=(__m128i& a, const int b) noexcept { return a = a << b; }
        inline __m128i& operator>>=(__m128i& a, const int b) noexcept { return a = a >> b; }

		[[nodiscard]] inline __m128i operator==(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpeq_epi64(a, b); }

        [[nodiscard]] inline __m128i operator>(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpgt_epi64(a, b); }
        [[nodiscard]] inline __m128i operator<(const __m128i& a, const __m128i& b) noexcept { return b > a; }
        [[nodiscard]] inline __m128i operator>=(const __m128i& a, const __m128i& b) noexcept { return ~(a < b); }
        [[nodiscard]] inline __m128i operator<=(const __m128i& a, const __m128i& b) noexcept { return ~(a > b); }

		[[nodiscard]] inline __m256i operator~(const __m256i& a) noexcept { return _mm256_andnot_si256(_mm256_set1_epi64x(0), a); }

        [[nodiscard]] inline __m256i operator+(const __m256i& a, const __m256i& b) noexcept { return _mm256_add_epi64(a, b); }
        [[nodiscard]] inline __m256i operator-(const __m256i& a, const __m256i& b) noexcept { return _mm256_sub_epi64(a, b); }
        [[nodiscard]] inline __m256i operator&(const __m256i& a, const __m256i& b) noexcept { return _mm256_and_si256(a, b); }
        [[nodiscard]] inline __m256i operator|(const __m256i& a, const __m256i& b) noexcept { return _mm256_or_si256(a, b); }
        [[nodiscard]] inline __m256i operator^(const __m256i& a, const __m256i& b) noexcept { return _mm256_xor_si256(a, b); }
        [[nodiscard]] inline __m256i operator<<(const __m256i& a, const int b) noexcept { return _mm256_slli_epi64(a, b); }
        [[nodiscard]] inline __m256i operator>>(const __m256i& a, const int b) noexcept { return _mm256_srli_epi64(a, b); }

        inline __m256i& operator+=(__m256i& a, const __m256i& b) noexcept { return a = a + b; }
        inline __m256i& operator-=(__m256i& a, const __m256i& b) noexcept { return a = a - b; }
        inline __m256i& operator&=(__m256i& a, const __m256i& b) noexcept { return a = a & b; }
        inline __m256i& operator|=(__m256i& a, const __m256i& b) noexcept { return a = a | b; }
        inline __m256i& operator^=(__m256i& a, const __m256i& b) noexcept { return a = a ^ b; }
        inline __m256i& operator<<=(__m256i& a, const int b) noexcept { return a = a << b; }
        inline __m256i& operator>>=(__m256i& a, const int b) noexcept { return a = a >> b; }

		uint64_t _mm256_reduce_or_epi64(__m256i) noexcept;
		uint32_t _mm256_reduce_add_epi32(__m256i) noexcept;
    #endif
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
