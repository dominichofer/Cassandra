#pragma once
#include "MacrosHell.h"
#include <cstdint>
#include <cstddef>

constexpr uint64_t Bit(const uint64_t position) noexcept
{
	assert(position < 64);
	return 1ui64 << position;
}

constexpr void SetBit(uint64_t& bit_field, const uint64_t position) noexcept
{
	assert(position < 64);
	bit_field |= Bit(position);
}

constexpr void ResetBit(uint64_t& bit_field, const uint64_t position) noexcept
{
	assert(position < 64);
	bit_field &= ~Bit(position);
}

constexpr bool TestBit(const uint64_t bit_field, const uint64_t position) noexcept
{
	assert(position < 64);
	return (bit_field & Bit(position)) != 0;
}

constexpr bool TestBits(const uint64_t bit_field, const uint64_t mask) noexcept
{
	return (bit_field & mask) == mask;
}

uint64_t FlipCodiagonal(uint64_t) noexcept;
uint64_t FlipDiagonal(uint64_t) noexcept;
uint64_t FlipHorizontal(uint64_t) noexcept;
uint64_t FlipVertical(uint64_t) noexcept;

inline unsigned int BitScanLSB(const uint64_t mask) noexcept
{
	// BitScanLSB(0) may be undefined.
	assert(mask);

	#if defined(_MSC_VER)
		unsigned long index = 0;
		_BitScanForward64(&index, mask);
		return index;
	#elif defined(__GNUC__)
		return __builtin_ctzll(mask); // __builtin_ctzll(0) is undefined
	#endif
}
	
inline unsigned int BitScanMSB(const uint64_t mask) noexcept
{
	// BitScanMSB(0) may be undefined.
	assert(mask);

	#if defined(_MSC_VER)
		unsigned long index = 0;
		_BitScanReverse64(&index, mask);
		return index;
	#elif defined(__GNUC__)
		return __builtin_clzll(mask) ^ 63; // __builtin_clzll(0) is undefined
	#endif
}

inline uint64_t CountLeadingZeros(const uint64_t mask) noexcept
{
	// CountLeadingZeros(0) may be undefined.
	assert(mask);

	#if defined(_MSC_VER)
		return _lzcnt_u64(mask); // _lzcnt_u64(0) == 64
	#elif defined(__GNUC__)
		return __builtin_clzll(mask); // __builtin_clzll(0) is undefined
	#endif
}

inline uint64_t CountTrailingZeros(const uint64_t mask) noexcept
{
	// CountTrailingZeros(0) may be undefined.
	assert(mask);

	#if defined(_MSC_VER)
			return _tzcnt_u64(mask); // _tzcnt_u64(0) is undefined
	#elif defined(__GNUC__)
			return __builtin_ctzll(mask); // __builtin_ctzll(0) is undefined
	#endif
}

inline uint64_t GetLSB(const uint64_t b) noexcept
{
	#ifdef HAS_BLSI
		return _blsi_u64(b); 
	#else
		#pragma warning(suppress : 4146)
		return b & -b;
	#endif
}

inline uint64_t GetMSB(const uint64_t b) noexcept
{
	if (b != 0u)
		return 0x8000000000000000ui64 >> CountLeadingZeros(b);
	return 0;
}

inline void RemoveLSB(uint64_t & b) noexcept
{
	#ifdef HAS_BLSR
		b = _blsr_u64(b);
	#else
		b &= b - 1;
	#endif
}

inline void RemoveMSB(uint64_t& b) noexcept
{
	b ^= GetMSB(b);
}

inline uint64_t PopCount(uint64_t b) noexcept
{
	#ifdef HAS_POPCNT
		#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
			return _mm_popcnt_u64(b);
		#else
			return __builtin_popcountll(b);
		#endif
	#else
		b -= (b >> 1) & 0x5555555555555555ui64;
		b = ((b >> 2) & 0x3333333333333333ui64) + (b & 0x3333333333333333ui64);
		b = ((b >> 4) + b) & 0x0F0F0F0F0F0F0F0Fui64;
		return (b * 0x0101010101010101ui64) >> 56;
	#endif
}

inline uint64_t BExtr(const uint64_t src, const unsigned int start, unsigned int len) noexcept
{
	#if defined(HAS_BEXTR) || defined(HAS_TBM)
		return _bextr_u64(src, start, len);
	#else
		return (src >> start) & ((1ui64 << len) - 1);
	#endif
}

inline uint64_t BZHI(const uint64_t src, const uint32_t index) noexcept
{
	#ifdef HAS_BZHI
		return _bzhi_u64(src, index);
	#else
		return src & ((1ui64 << index) - 1);
	#endif
}


inline uint64_t PDep(uint64_t src, uint64_t mask) noexcept
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

inline uint64_t PExt(uint64_t src, uint64_t mask) noexcept
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

inline uint64_t BSwap(const uint64_t b) noexcept
{
	#if defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}

#if defined(_MSC_VER)
    #ifdef HAS_SSE2
        inline __m128i operator~(const __m128i& a) noexcept { return _mm_andnot_si128(a, _mm_set1_epi64x(0xFFFFFFFFFFFFFFFFui64)); }

        inline __m128i operator+(const __m128i& a, const __m128i& b) noexcept { return _mm_add_epi64(a, b); }
        inline __m128i operator-(const __m128i& a, const __m128i& b) noexcept { return _mm_sub_epi64(a, b); }
        inline __m128i operator&(const __m128i& a, const __m128i& b) noexcept { return _mm_and_si128(a, b); }
        inline __m128i operator|(const __m128i& a, const __m128i& b) noexcept { return _mm_or_si128(a, b); }
        inline __m128i operator^(const __m128i& a, const __m128i& b) noexcept { return _mm_xor_si128(a, b); }
        inline __m128i operator<<(const __m128i& a, const int b) noexcept { return _mm_slli_epi64(a, b); }
        inline __m128i operator>>(const __m128i& a, const int b) noexcept { return _mm_srli_epi64(a, b); }

        inline __m128i operator+=(__m128i& a, const __m128i& b) noexcept { return a = a + b; }
        inline __m128i operator-=(__m128i& a, const __m128i& b) noexcept { return a = a - b; }
        inline __m128i operator&=(__m128i& a, const __m128i& b) noexcept { return a = a & b; }
        inline __m128i operator|=(__m128i& a, const __m128i& b) noexcept { return a = a | b; }
        inline __m128i operator^=(__m128i& a, const __m128i& b) noexcept { return a = a ^ b; }
        inline __m128i operator<<=(__m128i& a, const int b) noexcept { return a = a << b; }
        inline __m128i operator>>=(__m128i& a, const int b) noexcept { return a = a >> b; }
    #endif
    #ifdef HAS_SSE4_1
        inline __m128i operator==(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpeq_epi64(a, b); }
    #endif
    #ifdef HAS_SSE4_2
        inline __m128i operator>(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpgt_epi64(a, b); }
        inline __m128i operator<(const __m128i& a, const __m128i& b) noexcept { return b > a; }
        inline __m128i operator>=(const __m128i& a, const __m128i& b) noexcept { return ~(a < b); }
        inline __m128i operator<=(const __m128i& a, const __m128i& b) noexcept { return ~(a > b); }
    #endif
    #ifdef HAS_AVX2
		inline __m256i operator~(const __m256i& a) noexcept { return _mm256_xor_si256(a, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFui64)); }

        inline __m256i operator+(const __m256i& a, const __m256i& b) noexcept { return _mm256_add_epi64(a, b); }
        inline __m256i operator-(const __m256i& a, const __m256i& b) noexcept { return _mm256_sub_epi64(a, b); }
        inline __m256i operator&(const __m256i& a, const __m256i& b) noexcept { return _mm256_and_si256(a, b); }
        inline __m256i operator|(const __m256i& a, const __m256i& b) noexcept { return _mm256_or_si256(a, b); }
        inline __m256i operator^(const __m256i& a, const __m256i& b) noexcept { return _mm256_xor_si256(a, b); }
        inline __m256i operator<<(const __m256i& a, const int b) noexcept { return _mm256_slli_epi64(a, b); }
        inline __m256i operator>>(const __m256i& a, const int b) noexcept { return _mm256_srli_epi64(a, b); }

        inline __m256i operator+=(__m256i& a, const __m256i& b) noexcept { return a = a + b; }
        inline __m256i operator-=(__m256i& a, const __m256i& b) noexcept { return a = a - b; }
        inline __m256i operator&=(__m256i& a, const __m256i& b) noexcept { return a = a & b; }
        inline __m256i operator|=(__m256i& a, const __m256i& b) noexcept { return a = a | b; }
        inline __m256i operator^=(__m256i& a, const __m256i& b) noexcept { return a = a ^ b; }
        inline __m256i operator<<=(__m256i& a, const int b) noexcept { return a = a << b; }
        inline __m256i operator>>=(__m256i& a, const int b) noexcept { return a = a >> b; }
    #endif
	#ifdef HAS_AVX512
		inline __m512i operator~(const __m512i& a) noexcept { return _mm512_xor_si512(a, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFui64)); }

		inline __m512i operator+(const __m512i& a, const __m512i& b) noexcept { return _mm512_add_epi64(a, b); }
		inline __m512i operator-(const __m512i& a, const __m512i& b) noexcept { return _mm512_sub_epi64(a, b); }
		inline __m512i operator&(const __m512i& a, const __m512i& b) noexcept { return _mm512_and_si512(a, b); }
		inline __m512i operator|(const __m512i& a, const __m512i& b) noexcept { return _mm512_or_si512(a, b); }
		inline __m512i operator^(const __m512i& a, const __m512i& b) noexcept { return _mm512_xor_si512(a, b); }
		inline __m512i operator<<(const __m512i& a, const int b) noexcept { return _mm512_slli_epi64(a, b); }
		inline __m512i operator>>(const __m512i& a, const int b) noexcept { return _mm512_srli_epi64(a, b); }

		inline __m512i operator+=(__m512i& a, const __m512i& b) noexcept { return a = a + b; }
		inline __m512i operator-=(__m512i& a, const __m512i& b) noexcept { return a = a - b; }
		inline __m512i operator&=(__m512i& a, const __m512i& b) noexcept { return a = a & b; }
		inline __m512i operator|=(__m512i& a, const __m512i& b) noexcept { return a = a | b; }
		inline __m512i operator^=(__m512i& a, const __m512i& b) noexcept { return a = a ^ b; }
		inline __m512i operator<<=(__m512i& a, const int b) noexcept { return a = a << b; }
		inline __m512i operator>>=(__m512i& a, const int b) noexcept { return a = a >> b; }
	#endif
#endif
