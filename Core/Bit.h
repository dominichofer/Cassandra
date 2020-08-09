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

#ifndef __AVX2__
	#error This code requires AVX2!
#endif

#if !(defined(_M_X64) || defined(__x86_64))
	#error This code requires x64!
#endif
#include <cstdint>
#include <bit>

using int8 = signed char;
using int16 = short;
using int32 = int;
using int64 = long long;

using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;
using uint64 = unsigned long long;

template <int mantissa, int exponent>
class fp;

// IEEE 754
using float16 = fp<10,5>;
using float32 = float;
using float64 = double;


using tf32 = fp<10,8>; // nvidia
using bfloat16 = fp<7,8>; // google


class int8x16;
class int8x32;

class int16x8;
class int16x16;

class int32x4;
class int32x8;

class int64x2;
class int64x4;

namespace detail
{
	template <class T>
	class int128
	{
	protected:
		__m128i reg{0};
	public:
		int128() noexcept = default;
		int128(__m128i v) noexcept : reg(v) {}

		[[nodiscard]] operator __m128i() const noexcept { return reg; }

		[[nodiscard]] T operator~() const noexcept { return andnot(reg, T{-1}); }
		friend [[nodiscard]] T operator-(T v) noexcept { return T{} - v; }

		[[nodiscard]] T operator&(T v) const noexcept { return _mm_and_si128(reg, v); }
		[[nodiscard]] T operator|(T v) const noexcept { return _mm_or_si128(reg, v); }
		[[nodiscard]] T operator^(T v) const noexcept { return _mm_xor_si128(reg, v); }
		friend [[nodiscard]] T andnot(T l, T r) noexcept { return _mm_andnot_si128(l, r); }

		T& operator+=(T v) noexcept { reg = static_cast<T&>(*this) + v; return static_cast<T&>(*this); }
		T& operator-=(T v) noexcept { reg = static_cast<T&>(*this) - v; return static_cast<T&>(*this); }
		T& operator&=(T v) noexcept { reg = static_cast<T&>(*this) & v; return static_cast<T&>(*this); }
		T& operator|=(T v) noexcept { reg = static_cast<T&>(*this) | v; return static_cast<T&>(*this); }
		T& operator^=(T v) noexcept { reg = static_cast<T&>(*this) ^ v; return static_cast<T&>(*this); }
		T& operator<<=(int v) noexcept { reg = static_cast<T&>(*this) << v; return static_cast<T&>(*this); }
		T& operator>>=(int v) noexcept { reg = static_cast<T&>(*this) >> v; return static_cast<T&>(*this); }
		T& operator<<=(T v) noexcept { reg = static_cast<T&>(*this) << v; return static_cast<T&>(*this); }
		T& operator>>=(T v) noexcept { reg = static_cast<T&>(*this) >> v; return static_cast<T&>(*this); }
	};

	template <class T>
	class int256
	{
	protected:
		__m256i reg{0};
	public:
		int256() noexcept = default;
		int256(__m256i v) noexcept : reg(v) {}

		[[nodiscard]] operator __m256i() const noexcept { return reg; }

		[[nodiscard]] T operator~() const noexcept { return andnot(reg, T{-1}); }
		friend [[nodiscard]] T operator-(T v) noexcept { return T{} - v; }

		[[nodiscard]] T operator&(T v) const noexcept { return _mm256_and_si256(reg, v); }
		[[nodiscard]] T operator|(T v) const noexcept { return _mm256_or_si256(reg, v); }
		[[nodiscard]] T operator^(T v) const noexcept { return _mm256_xor_si256(reg, v); }
		friend [[nodiscard]] T andnot(T l, T r) noexcept { return _mm256_andnot_si256(l, r); }

		T& operator+=(T v) noexcept { reg = static_cast<T&>(*this) + v; return static_cast<T&>(*this); }
		T& operator-=(T v) noexcept { reg = static_cast<T&>(*this) - v; return static_cast<T&>(*this); }
		T& operator&=(T v) noexcept { reg = static_cast<T&>(*this) & v; return static_cast<T&>(*this); }
		T& operator|=(T v) noexcept { reg = static_cast<T&>(*this) | v; return static_cast<T&>(*this); }
		T& operator^=(T v) noexcept { reg = static_cast<T&>(*this) ^ v; return static_cast<T&>(*this); }
		T& operator<<=(int v) noexcept { reg = static_cast<T&>(*this) << v; return static_cast<T&>(*this); }
		T& operator>>=(int v) noexcept { reg = static_cast<T&>(*this) >> v; return static_cast<T&>(*this); }
		T& operator<<=(T v) noexcept { reg = static_cast<T&>(*this) << v; return static_cast<T&>(*this); }
		T& operator>>=(T v) noexcept { reg = static_cast<T&>(*this) >> v; return static_cast<T&>(*this); }
	};
}

class int64x2 : public detail::int128<int64x2>
{
public:
	int64x2(__m128i v) noexcept : detail::int128<int64x2>(v) {}
	int64x2() noexcept = default;
	int64x2(int64 v) noexcept : int64x2(_mm_set1_epi64x(v)) {}
	explicit int64x2(int64 a, int64 b) noexcept : int64x2(_mm_set_epi64x(a,b)) {}

	[[nodiscard]] explicit operator int64() const noexcept { return _mm_cvtsi128_si64(reg); }

	[[nodiscard]] int64x2 operator+(int64x2 v) const noexcept { return _mm_add_epi64(reg, v); }
	[[nodiscard]] int64x2 operator-(int64x2 v) const noexcept { return _mm_sub_epi64(reg, v); }
	[[nodiscard]] int64x2 operator<<(int v) const noexcept { return _mm_slli_epi64(reg, v); }
	[[nodiscard]] int64x2 operator>>(int v) const noexcept { return _mm_srli_epi64(reg, v); }
	[[nodiscard]] int64x2 operator<<(int64x2 v) const noexcept { return _mm_sllv_epi64(reg, v); }
	[[nodiscard]] int64x2 operator>>(int64x2 v) const noexcept { return _mm_srlv_epi64(reg, v); }

	[[nodiscard]] int64x2 operator==(int64x2 v) const noexcept { return _mm_cmpeq_epi64(reg, v); }
	[[nodiscard]] int64x2 operator!=(int64x2 v) const noexcept { return ~(*this == v); }
	[[nodiscard]] int64x2 operator> (int64x2 v) const noexcept { return _mm_cmpgt_epi64(reg, v); }
	[[nodiscard]] int64x2 operator< (int64x2 v) const noexcept { return _mm_cmpgt_epi64(v, reg); }
	[[nodiscard]] int64x2 operator>=(int64x2 v) const noexcept { return ~(*this < v); }
	[[nodiscard]] int64x2 operator<=(int64x2 v) const noexcept { return ~(*this > v); }
};

[[nodiscard]] inline int64x2 unpackhi(int64x2 a, int64x2 b) noexcept { return _mm_unpackhi_epi64(a, b); }


class int64x4 : public detail::int256<int64x4>
{
public:
	int64x4(__m256i v) noexcept : detail::int256<int64x4>(v) {}
	int64x4() noexcept = default;
	int64x4(int64 v) noexcept : int64x4(_mm256_set1_epi64x(v)) {}
	explicit int64x4(int64 a, int64 b, int64 c, int64 d) noexcept : int64x4(_mm256_set_epi64x(a,b,c,d)) {}

	[[nodiscard]] explicit operator int64x2() const noexcept { return _mm256_castsi256_si128(reg); /*nop*/ }
	[[nodiscard]] explicit operator int64() const noexcept { return _mm_cvtsi128_si64(_mm256_castsi256_si128(reg)); }

	[[nodiscard]] int64x4 operator+(int64x4 v) const noexcept { return _mm256_add_epi64(reg, v); }
	[[nodiscard]] int64x4 operator-(int64x4 v) const noexcept { return _mm256_sub_epi64(reg, v); }
	[[nodiscard]] int64x4 operator<<(int v) const noexcept { return _mm256_slli_epi64(reg, v); }
	[[nodiscard]] int64x4 operator>>(int v) const noexcept { return _mm256_srli_epi64(reg, v); }
	[[nodiscard]] int64x4 operator<<(int64x4 v) const noexcept { return _mm256_sllv_epi64(reg, v); }
	[[nodiscard]] int64x4 operator>>(int64x4 v) const noexcept { return _mm256_srlv_epi64(reg, v); }

	[[nodiscard]] int64x4 operator==(int64x4 v) const noexcept { return _mm256_cmpeq_epi64(reg, v); }
	[[nodiscard]] int64x4 operator!=(int64x4 v) const noexcept { return ~(*this == v); }
	[[nodiscard]] int64x4 operator> (int64x4 v) const noexcept { return _mm256_cmpgt_epi64(reg, v); }
	[[nodiscard]] int64x4 operator< (int64x4 v) const noexcept { return _mm256_cmpgt_epi64(v, reg); }
	[[nodiscard]] int64x4 operator>=(int64x4 v) const noexcept { return ~(*this < v); }
	[[nodiscard]] int64x4 operator<=(int64x4 v) const noexcept { return ~(*this > v); }
};

[[nodiscard]] inline int64 reduce_or(int64x4 v) noexcept
{
	int64x2 y = static_cast<int64x2>(v).operator|(_mm256_extracti128_si256(v, 1));
	y |= unpackhi(y, y);
	return static_cast<int64>(y);
}


// TODO: Rename!
//[[nodiscard]]
//uint64_t Horizontal(int i) noexcept { return 0xFFULL << (i * 8); }
//
//[[nodiscard]]
//uint64_t Vertical(int i) noexcept { return 0x0101010101010101ULL << i; }
//
//[[nodiscard]]
//uint64_t Diagonal(int i) noexcept
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
	return __popcnt64(b);
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
