#pragma once
// Predefined macros:
// __GNUC__           Compiler is gcc.
// __clang__          Compiler is clang.
// __INTEL_COMPILER   Compiler is Intel.
// _MSC_VER           Compiler is Microsoft Visual Studio.
// _M_X64             Microsoft specific macro when targeting 64 bit based machines.
// __x86_64           Defined by GNU C and Sun Studio when targeting 64 bit based machines.

#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(__GNUC__)
	#include <x86intrin.h>
#else
	#error compiler not supported!
#endif

#if !(defined(_M_X64) || defined(__x86_64))
	#error This code requires x64!
#endif

using uint = unsigned int;

using int8 = signed char;
using int16 = signed short;
using int32 = signed int;
using int64 = signed long long;

using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;
using uint64 = unsigned long long;

constexpr int8 operator""_i8(unsigned long long v) { return static_cast<int8>(v); }
constexpr int16 operator""_i16(unsigned long long v) { return static_cast<int16>(v); }
constexpr int32 operator""_i32(unsigned long long v) { return static_cast<int32>(v); }
constexpr int64 operator""_i64(unsigned long long v) { return static_cast<int64>(v); }

constexpr uint8 operator""_ui8(unsigned long long v) { return static_cast<uint8>(v); }
constexpr uint16 operator""_ui16(unsigned long long v) { return static_cast<uint16>(v); }
constexpr uint32 operator""_ui32(unsigned long long v) { return static_cast<uint32>(v); }
constexpr uint64 operator""_ui64(unsigned long long v) { return static_cast<uint64>(v); }


#ifdef __AVX2__

// forward declarations
class int128;
int128 cmpeq_64(int128, int128) noexcept;
int64 reduce_and_64(int128) noexcept;
class int256;
int256 cmpeq_64(int256, int256) noexcept;
int64 reduce_and_64(int256) noexcept;

class int128
{
protected:
	__m128i reg{ 0 };
public:
	int128() = default;
	int128(__m128i v) noexcept : reg(v) {}
	explicit int128(std::nullptr_t) noexcept = delete;
	explicit int128(const int64* p) noexcept : reg(_mm_load_si128(reinterpret_cast<const __m128i*>(p))) {}
	explicit int128(const int32* p) noexcept : reg(_mm_load_si128(reinterpret_cast<const __m128i*>(p))) {}
	explicit int128(const int16* p) noexcept : reg(_mm_load_si128(reinterpret_cast<const __m128i*>(p))) {}
	explicit int128(const int8* p) noexcept : reg(_mm_load_si128(reinterpret_cast<const __m128i*>(p))) {}
	explicit int128(int64 v) noexcept : reg(_mm_set1_epi64x(v)) {}
	explicit int128(int32 v) noexcept : reg(_mm_set1_epi32(v)) {}
	explicit int128(int16 v) noexcept : reg(_mm_set1_epi16(v)) {}
	explicit int128(int8 v) noexcept : reg(_mm_set1_epi8(v)) {}
	int128(int64 e1, int64 e0) noexcept : reg(_mm_set_epi64x(e1, e0)) {}
	int128(int32 e3, int32 e2, int32 e1, int32 e0) noexcept : reg(_mm_set_epi32(e3, e2, e1, e0)) {}
	int128(int16 e7, int16 e6, int16 e5, int16 e4, int16 e3, int16 e2, int16 e1, int16 e0) noexcept : reg(_mm_set_epi16(e7, e6, e5, e4, e3, e2, e1, e0)) {}
	int128(int8 e15, int8 e14, int8 e13, int8 e12, int8 e11, int8 e10, int8 e9, int8 e8, int8 e7, int8 e6, int8 e5, int8 e4, int8 e3, int8 e2, int8 e1, int8 e0) noexcept : reg(_mm_set_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)) {}

	operator __m128i() const noexcept { return reg; }

	template <uint index> int8  get_int8 () const noexcept { static_assert(index < 16); return _mm_extract_epi8 (reg, index); }
	template <uint index> int16 get_int16() const noexcept { static_assert(index <  8); return _mm_extract_epi16(reg, index); }
	template <uint index> int32 get_int32() const noexcept { static_assert(index <  4); return _mm_extract_epi32(reg, index); }
	template <uint index> int64 get_int64() const noexcept { static_assert(index <  2); return _mm_extract_epi64(reg, index); }

	template <uint index> void set_int8 (int8  v) noexcept { static_assert(index < 16); reg = _mm_insert_epi8(reg, v, index); }
	template <uint index> void set_int16(int16 v) noexcept { static_assert(index < 8); reg = _mm_insert_epi16(reg, v, index); }
	template <uint index> void set_int32(int32 v) noexcept { static_assert(index < 4); reg = _mm_insert_epi32(reg, v, index); }
	template <uint index> void set_int64(int64 v) noexcept { static_assert(index < 2); reg = _mm_insert_epi64(reg, v, index); }

	template <uint index> int128 with_int8 (int8  v) noexcept { static_assert(index < 16); return _mm_insert_epi8(reg, v, index); }
	template <uint index> int128 with_int16(int16 v) noexcept { static_assert(index < 8); return _mm_insert_epi16(reg, v, index); }
	template <uint index> int128 with_int32(int32 v) noexcept { static_assert(index < 4); return _mm_insert_epi32(reg, v, index); }
	template <uint index> int128 with_int64(int64 v) noexcept { static_assert(index < 2); return _mm_insert_epi64(reg, v, index); }

	bool operator==(int128 o) const noexcept { auto x = _mm_xor_si128(reg, o.reg); return _mm_testz_si128(x, x); }
	bool operator!=(int128 o) const noexcept { return !(*this == o); }

	int128 operator~() const noexcept { return andnot(reg, int128{ -1 }); }
	int128 operator&(int128 v) const noexcept { return _mm_and_si128(reg, v); }
	int128 operator|(int128 v) const noexcept { return _mm_or_si128(reg, v); }
	int128 operator^(int128 v) const noexcept { return _mm_xor_si128(reg, v); }
	int128& operator&=(int128 v) noexcept { reg = *this & v; return *this; }
	int128& operator|=(int128 v) noexcept { reg = *this | v; return *this; }
	int128& operator^=(int128 v) noexcept { reg = *this ^ v; return *this; }
	friend int128 andnot(int128 l, int128 r) noexcept { return _mm_andnot_si128(l, r); }
};

class int64x2 : public int128
{
public:
	int64x2() = default;
	int64x2(__m128i v) noexcept : int128(v) {}
	explicit int64x2(std::nullptr_t) noexcept = delete;
	explicit int64x2(const int64 * p) noexcept : int128(p) {}
	explicit int64x2(int64 v) noexcept : int128(v) {}
	int64x2(int64 e1, int64 e0) noexcept : int128(e1, e0) {}
	
	template <uint index> int64 get() const noexcept { return get_int64<index>(); }
	template <uint index> void set(int64 v) noexcept { set_int64<index>(); }
	template <uint index> int128 with(int64 v) noexcept { return with_int64<index>(v); }

	int64x2 operator~() const noexcept { return andnot(reg, int64x2{ -1 }); }
	int64x2 operator-() const noexcept { return int64x2{} - *this; }
	int64x2 operator&(int64x2 v) const noexcept { return _mm_and_si128(reg, v); }
	int64x2 operator|(int64x2 v) const noexcept { return _mm_or_si128(reg, v); }
	int64x2 operator^(int64x2 v) const noexcept { return _mm_xor_si128(reg, v); }
	int64x2 operator+(int64x2 v) const noexcept { return _mm_add_epi64(reg, v.reg); }
	int64x2 operator-(int64x2 v) const noexcept { return _mm_sub_epi64(reg, v.reg); }
	//int64x2 operator*(int64x2 v) const noexcept { return _mm_mul_epi64(reg, v.reg); } // Instruction does not exist.
	// int64x2 operator/(int64x2 v) const noexcept { return _mm_div_epi64(reg, v.reg); }
	// int64x2 operator%(int64x2 v) const noexcept { return _mm_rem_epi64(reg, v.reg); }
	int64x2 operator<<(int64x2 v) const noexcept { return _mm_sllv_epi64(reg, v.reg); }
	int64x2 operator>>(int64x2 v) const noexcept { return _mm_srlv_epi64(reg, v.reg); }
	int64x2 operator<<(int v) const noexcept { return _mm_slli_epi64(reg, v); }
	int64x2 operator>>(int v) const noexcept { return _mm_srli_epi64(reg, v); }
	int64x2& operator&=(int64x2 v) noexcept { reg = *this & v; return *this; }
	int64x2& operator|=(int64x2 v) noexcept { reg = *this | v; return *this; }
	int64x2& operator^=(int64x2 v) noexcept { reg = *this ^ v; return *this; }
	friend int64x2 andnot(int64x2 l, int64x2 r) noexcept { return _mm_andnot_si128(l, r); }

	int64x2& operator+=(int64x2 v) noexcept { reg = *this + v; return *this; }
	int64x2& operator-=(int64x2 v) noexcept { reg = *this - v; return *this; }
	//int64x2& operator*=(int64x2 v) noexcept { reg = *this * v; return *this; } // Instruction does not exist.
	// int64x2& operator/=(int64x2 v) noexcept { reg = *this / v; return *this; }
	int64x2& operator<<=(int64x2 v) noexcept { reg = *this << v; return *this; }
	int64x2& operator>>=(int64x2 v) noexcept { reg = *this >> v; return *this; }
	int64x2& operator<<=(int v) noexcept { reg = *this << v; return *this; }
	int64x2& operator>>=(int v) noexcept { reg = *this >> v; return *this; }
};

class int32x4 : public int128
{
public:
	int32x4() = default;
	int32x4(__m128i v) noexcept : int128(v) {}
	explicit int32x4(std::nullptr_t) noexcept = delete;
	explicit int32x4(const int32* p) noexcept : int128(p) {}
	explicit int32x4(int32 v) noexcept : int128(v) {}
	int32x4(int32 e3, int32 e2, int32 e1, int32 e0) noexcept : int128(e3, e2, e1, e0) {}

	template <uint index> int32 get() const noexcept { return get_int32<index>(); }
	template <uint index> void set(int32 v) noexcept { set_int32<index>(); }
	template <uint index> int128 with(int32 v) noexcept { return with_int32<index>(v); }

	int32x4 operator~() const noexcept { return andnot(reg, int32x4{ -1 }); }
	int32x4 operator-() const noexcept { return int32x4{} - *this; }
	int32x4 operator&(int32x4 v) const noexcept { return _mm_and_si128(reg, v); }
	int32x4 operator|(int32x4 v) const noexcept { return _mm_or_si128(reg, v); }
	int32x4 operator^(int32x4 v) const noexcept { return _mm_xor_si128(reg, v); }
	int32x4 operator+(int32x4 v) const noexcept { return _mm_add_epi32(reg, v.reg); }
	int32x4 operator-(int32x4 v) const noexcept { return _mm_sub_epi32(reg, v.reg); }
	int32x4 operator*(int32x4 v) const noexcept { return _mm_mul_epi32(reg, v.reg); }
	// int32x4 operator/(int32x4 v) const noexcept { return _mm_div_epi32(reg, v.reg); }
	// int32x4 operator%(int32x4 v) const noexcept { return _mm_rem_epi32(reg, v.reg); }
	int32x4 operator<<(int32x4 v) const noexcept { return _mm_sllv_epi32(reg, v.reg); }
	int32x4 operator>>(int32x4 v) const noexcept { return _mm_srlv_epi32(reg, v.reg); }
	int32x4 operator<<(int v) const noexcept { return _mm_slli_epi32(reg, v); }
	int32x4 operator>>(int v) const noexcept { return _mm_srli_epi32(reg, v); }
	int32x4& operator&=(int32x4 v) noexcept { reg = *this & v; return *this; }
	int32x4& operator|=(int32x4 v) noexcept { reg = *this | v; return *this; }
	int32x4& operator^=(int32x4 v) noexcept { reg = *this ^ v; return *this; }
	friend int32x4 andnot(int32x4 l, int32x4 r) noexcept { return _mm_andnot_si128(l, r); }

	int32x4& operator+=(int32x4 v) noexcept { reg = *this + v; return *this; }
	int32x4& operator-=(int32x4 v) noexcept { reg = *this - v; return *this; }
	int32x4& operator*=(int32x4 v) noexcept { reg = *this * v; return *this; }
	// int32x4& operator/=(int32x4 v) noexcept { reg = *this / v; return *this; }
	int32x4& operator<<=(int32x4 v) noexcept { reg = *this << v; return *this; }
	int32x4& operator>>=(int32x4 v) noexcept { reg = *this >> v; return *this; }
	int32x4& operator<<=(int v) noexcept { reg = *this << v; return *this; }
	int32x4& operator>>=(int v) noexcept { reg = *this >> v; return *this; }
};

class int16x8 : public int128
{
public:
	int16x8() = default;
	int16x8(__m128i v) noexcept : int128(v) {}
	explicit int16x8(std::nullptr_t) noexcept = delete;
	explicit int16x8(const int16* p) noexcept : int128(p) {}
	explicit int16x8(int16 v) noexcept : int128(v) {}
	int16x8(int16 e7, int16 e6, int16 e5, int16 e4, int16 e3, int16 e2, int16 e1, int16 e0) noexcept : int128(e7, e6, e5, e4, e3, e2, e1, e0) {}

	template <uint index> int16 get() const noexcept { return get_int16<index>(); }
	template <uint index> void set(int16 v) noexcept { set_int16<index>(); }
	template <uint index> int128 with(int16 v) noexcept { return with_int16<index>(v); }

	int16x8 operator~() const noexcept { return andnot(reg, int16x8{ -1 }); }
	int16x8 operator-() const noexcept { return int16x8{} - *this; }
	int16x8 operator&(int16x8 v) const noexcept { return _mm_and_si128(reg, v); }
	int16x8 operator|(int16x8 v) const noexcept { return _mm_or_si128(reg, v); }
	int16x8 operator^(int16x8 v) const noexcept { return _mm_xor_si128(reg, v); }
	int16x8 operator+(int16x8 v) const noexcept { return _mm_add_epi16(reg, v.reg); }
	int16x8 operator-(int16x8 v) const noexcept { return _mm_sub_epi16(reg, v.reg); }
	//int16x8 operator*(int16x8 v) const noexcept { return _mm_mul_epi16(reg, v.reg); } // Instruction does not exist.
	// int16x8 operator/(int16x8 v) const noexcept { return _mm_div_epi16(reg, v.reg); }
	// int16x8 operator%(int16x8 v) const noexcept { return _mm_rem_epi16(reg, v.reg); }
	//int16x8 operator<<(int16x8 v) const noexcept { return _mm_sllv_epi16(reg, v.reg); } // Instruction requires AVX512VL and AVX512BW.
	//int16x8 operator>>(int16x8 v) const noexcept { return _mm_srlv_epi16(reg, v.reg); } // Instruction requires AVX512VL and AVX512BW.
	int16x8 operator<<(int v) const noexcept { return _mm_slli_epi16(reg, v); }
	int16x8 operator>>(int v) const noexcept { return _mm_srli_epi16(reg, v); }
	int16x8& operator&=(int16x8 v) noexcept { reg = *this & v; return *this; }
	int16x8& operator|=(int16x8 v) noexcept { reg = *this | v; return *this; }
	int16x8& operator^=(int16x8 v) noexcept { reg = *this ^ v; return *this; }
	friend int16x8 andnot(int16x8 l, int16x8 r) noexcept { return _mm_andnot_si128(l, r); }

	int16x8& operator+=(int16x8 v) noexcept { reg = *this + v; return *this; }
	int16x8& operator-=(int16x8 v) noexcept { reg = *this - v; return *this; }
	//int16x8& operator*=(int16x8 v) noexcept { reg = *this * v; return *this; } // Instruction does not exist.
	// int16x8& operator/=(int16x8 v) noexcept { reg = *this / v; return *this; }
	//int16x8& operator<<=(int16x8 v) noexcept { reg = *this << v; return *this; } // Instruction requires AVX512VL and AVX512BW.
	//int16x8& operator>>=(int16x8 v) noexcept { reg = *this >> v; return *this; } // Instruction requires AVX512VL and AVX512BW.
	int16x8& operator<<=(int v) noexcept { reg = *this << v; return *this; }
	int16x8& operator>>=(int v) noexcept { reg = *this >> v; return *this; }
};

class int8x16 : public int128
{
public:
	int8x16() = default;
	int8x16(__m128i v) noexcept : int128(v) {}
	explicit int8x16(std::nullptr_t) noexcept = delete;
	explicit int8x16(const int8* p) noexcept : int128(p) {}
	explicit int8x16(int8 v) noexcept : int128(v) {}
	int8x16(int8 e15, int8 e14, int8 e13, int8 e12, int8 e11, int8 e10, int8 e9, int8 e8,
		int8 e7, int8 e6, int8 e5, int8 e4, int8 e3, int8 e2, int8 e1, int8 e0) noexcept
		: int128(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) {}

	template <uint index> int8 get() const noexcept { return get_int8<index>(); }
	template <uint index> void set(int8 v) noexcept { set_int8<index>(); }
	template <uint index> int128 with(int8 v) noexcept { return with_int8<index>(v); }

	int8x16 operator~() const noexcept { return andnot(reg, int8x16{ -1 }); }
	int8x16 operator-() const noexcept { return int8x16{} - *this; }
	int8x16 operator&(int8x16 v) const noexcept { return _mm_and_si128(reg, v); }
	int8x16 operator|(int8x16 v) const noexcept { return _mm_or_si128(reg, v); }
	int8x16 operator^(int8x16 v) const noexcept { return _mm_xor_si128(reg, v); }
	int8x16 operator+(int8x16 v) const noexcept { return _mm_add_epi8(reg, v.reg); }
	int8x16 operator-(int8x16 v) const noexcept { return _mm_sub_epi8(reg, v.reg); }
	//int8x16 operator*(int8x16 v) const noexcept { return _mm_mul_epi8(reg, v.reg); } // Instruction does not exist.
	// int8x16 operator/(int8x16 v) const noexcept { return _mm_div_epi8(reg, v.reg); }
	// int8x16 operator%(int8x16 v) const noexcept { return _mm_rem_epi8(reg, v.reg); }
	//int8x16 operator<<(int8x16 v) const noexcept { return _mm_sllv_epi8(reg, v.reg); } // Instruction does not exist.
	//int8x16 operator>>(int8x16 v) const noexcept { return _mm_srlv_epi8(reg, v.reg); } // Instruction does not exist.
	//int8x16 operator<<(int v) const noexcept { return _mm_slli_epi8(reg, v); } // Instruction does not exist.
	//int8x16 operator>>(int v) const noexcept { return _mm_srli_epi8(reg, v); } // Instruction does not exist.
	int8x16& operator&=(int8x16 v) noexcept { reg = *this & v; return *this; }
	int8x16& operator|=(int8x16 v) noexcept { reg = *this | v; return *this; }
	int8x16& operator^=(int8x16 v) noexcept { reg = *this ^ v; return *this; }
	friend int8x16 andnot(int8x16 l, int8x16 r) noexcept { return _mm_andnot_si128(l, r); }

	int8x16& operator+=(int8x16 v) noexcept { reg = *this + v; return *this; }
	int8x16& operator-=(int8x16 v) noexcept { reg = *this - v; return *this; }
	//int8x16& operator*=(int8x16 v) noexcept { reg = *this * v; return *this; } // Instruction does not exist.
	// int8x16& operator/=(int8x16 v) noexcept { reg = *this / v; return *this; }
	//int8x16& operator<<=(int8x16 v) noexcept { reg = *this << v; return *this; } // Instruction does not exist.
	//int8x16& operator>>=(int8x16 v) noexcept { reg = *this >> v; return *this; } // Instruction does not exist.
	//int8x16& operator<<=(int v) noexcept { reg = *this << v; return *this; } // Instruction does not exist.
	//int8x16& operator>>=(int v) noexcept { reg = *this >> v; return *this; } // Instruction does not exist.
};


// unpackhi({a,_}, {c,_}) -> {c,a}
inline int128 unpackhi_64(int128 x, int128 y) noexcept { return _mm_unpackhi_epi64(x, y); }
inline int64x2 unpackhi(int64x2 x, int64x2 y) noexcept { return _mm_unpackhi_epi64(x, y); }

// unpacklo({_,b}, {_,d}) -> {d,b}
inline int128 unpacklo_64(int128 x, int128 y) noexcept { return _mm_unpacklo_epi64(x, y); }
inline int64x2 unpacklo(int64x2 x, int64x2 y) noexcept { return _mm_unpacklo_epi64(x, y); }

// unpackhi({a,b,_,_}, {e,f,_,_}) -> {e,a,f,b}
inline int128 unpackhi_32(int128 x, int128 y) noexcept { return _mm_unpackhi_epi32(x, y); }
inline int32x4 unpackhi(int32x4 x, int32x4 y) noexcept { return _mm_unpackhi_epi32(x, y); }

// unpacklo({_,_,c,d}, {_,_,g,h}) -> {g,c,h,d}
inline int128 unpacklo_32(int128 x, int128 y) noexcept { return _mm_unpacklo_epi32(x, y); }
inline int32x4 unpacklo(int32x4 x, int32x4 y) noexcept { return _mm_unpacklo_epi32(x, y); }

// unpackhi({a,b,c,d,_,_,_,_}, {i,j,k,l,_,_,_,_}) -> {i,a,j,b,k,c,l,d}
inline int128 unpackhi_16(int128 x, int128 y) noexcept { return _mm_unpackhi_epi16(x, y); }
inline int16x8 unpackhi(int16x8 x, int16x8 y) noexcept { return _mm_unpackhi_epi16(x, y); }

// unpacklo({_,_,_,_,e,f,g,h}, {_,_,_,_,m,n,o,p})) -> {m,e,n,f,o,g,p,h}
inline int128 unpacklo_16(int128 x, int128 y) noexcept { return _mm_unpacklo_epi16(x, y); }
inline int16x8 unpacklo(int16x8 x, int16x8 y) noexcept { return _mm_unpacklo_epi16(x, y); }

// unpackhi({a,b,c,d,e,f,g,h,_,_,_,_,_,_,_,_}, {a2,b2,c2,d2,e2,f2,g2,h2,_,_,_,_,_,_,_,_}) -> {a2,a,b2,b,c2,c,d2,d,e2,e,f2,f,g2,g,h2,h}
inline int128 unpackhi_8(int128 x, int128 y) noexcept { return _mm_unpackhi_epi8(x, y); }
inline int8x16 unpackhi(int8x16 x, int8x16 y) noexcept { return _mm_unpackhi_epi8(x, y); }

// unpacklo({_,_,_,_,_,_,_,_,i,j,k,l,m,n,o,p}, {_,_,_,_,_,_,_,_,i2,j2,k2,l2,m2,n2,o2,p2})) -> {i2,i,j2,j,k2,k,l2,l,m2,m,n2,n,o2,o,p2,p}
inline int128 unpacklo_8(int128 x, int128 y) noexcept { return _mm_unpacklo_epi8(x, y); }
inline int8x16 unpacklo(int8x16 x, int8x16 y) noexcept { return _mm_unpacklo_epi8(x, y); }

// returns a==b ? -1 : 0
inline int128 cmpeq_64(int128 a, int128 b) noexcept { return _mm_cmpeq_epi64(a, b); }
inline int128 cmpeq_32(int128 a, int128 b) noexcept { return _mm_cmpeq_epi32(a, b); }
inline int128 cmpeq_16(int128 a, int128 b) noexcept { return _mm_cmpeq_epi16(a, b); }
inline int128 cmpeq_8 (int128 a, int128 b) noexcept { return _mm_cmpeq_epi8 (a, b); }
inline int64x2 cmpeq(int64x2 a, int64x2 b) noexcept { return _mm_cmpeq_epi64(a, b); }
inline int32x4 cmpeq(int32x4 a, int32x4 b) noexcept { return _mm_cmpeq_epi32(a, b); }
inline int16x8 cmpeq(int16x8 a, int16x8 b) noexcept { return _mm_cmpeq_epi16(a, b); }
inline int8x16 cmpeq(int8x16 a, int8x16 b) noexcept { return _mm_cmpeq_epi8 (a, b); }

// returns a>b ? -1 : 0
inline int128 cmpgt_64(int128 a, int128 b) noexcept { return _mm_cmpgt_epi64(a, b); }
inline int128 cmpgt_32(int128 a, int128 b) noexcept { return _mm_cmpgt_epi32(a, b); }
inline int128 cmpgt_16(int128 a, int128 b) noexcept { return _mm_cmpgt_epi16(a, b); }
inline int128 cmpgt_8 (int128 a, int128 b) noexcept { return _mm_cmpgt_epi8 (a, b); }
inline int64x2 cmpgt(int64x2 a, int64x2 b) noexcept { return _mm_cmpgt_epi64(a, b); }
inline int32x4 cmpgt(int32x4 a, int32x4 b) noexcept { return _mm_cmpgt_epi32(a, b); }
inline int16x8 cmpgt(int16x8 a, int16x8 b) noexcept { return _mm_cmpgt_epi16(a, b); }
inline int8x16 cmpgt(int8x16 a, int8x16 b) noexcept { return _mm_cmpgt_epi8 (a, b); }

//inline int128 abs_64(int128 v) noexcept { return _mm_abs_epi64(v); } // Instruction requires AVX512VL and AVX512F.
inline int128 abs_32(int128 v) noexcept { return _mm_abs_epi32(v); }
inline int128 abs_16(int128 v) noexcept { return _mm_abs_epi16(v); }
inline int128 abs_8 (int128 v) noexcept { return _mm_abs_epi8 (v); }
//inline int64x2 abs(int64x2 v) noexcept { return _mm_abs_epi64(v); } // Instruction requires AVX512VL and AVX512F.
inline int32x4 abs(int32x4 v) noexcept { return _mm_abs_epi32(v); }
inline int16x8 abs(int16x8 v) noexcept { return _mm_abs_epi16(v); }
inline int8x16 abs(int8x16 v) noexcept { return _mm_abs_epi8 (v); }

inline int64 reduce_and_64(int128 v) noexcept { return v.get_int64<0>() & v.get_int64<1>(); }
inline int64 reduce_or_64 (int128 v) noexcept { return v.get_int64<0>() | v.get_int64<1>(); }
inline int64 reduce_xor_64(int128 v) noexcept { return v.get_int64<0>() ^ v.get_int64<1>(); }
inline int64 reduce_add_64(int128 v) noexcept { return v.get_int64<0>() + v.get_int64<1>(); }
inline int32 reduce_and_32(int128 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t & (t >> 32)); }
inline int32 reduce_or_32 (int128 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t | (t >> 32)); }
inline int32 reduce_xor_32(int128 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t ^ (t >> 32)); }
inline int32 reduce_add_32(int128 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t + (t >> 32)); }
inline int16 reduce_and_16(int128 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t & (t >> 16)); }
inline int16 reduce_or_16 (int128 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t | (t >> 16)); }
inline int16 reduce_xor_16(int128 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t ^ (t >> 16)); }
inline int16 reduce_add_16(int128 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t + (t >> 16)); }
inline int8  reduce_and_8 (int128 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t & (t >> 8)); }
inline int8  reduce_or_8  (int128 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t | (t >> 8)); }
inline int8  reduce_xor_8 (int128 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t ^ (t >> 8)); }
inline int8  reduce_add_8 (int128 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t + (t >> 8)); }
inline int64 reduce_and(int64x2 v) noexcept { return v.get_int64<0>() & v.get_int64<1>(); }
inline int64 reduce_or (int64x2 v) noexcept { return v.get_int64<0>() | v.get_int64<1>(); }
inline int64 reduce_xor(int64x2 v) noexcept { return v.get_int64<0>() ^ v.get_int64<1>(); }
inline int64 reduce_add(int64x2 v) noexcept { return v.get_int64<0>() + v.get_int64<1>(); }
inline int32 reduce_and(int32x4 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t & (t >> 32)); }
inline int32 reduce_or (int32x4 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t | (t >> 32)); }
inline int32 reduce_xor(int32x4 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t ^ (t >> 32)); }
inline int32 reduce_add(int32x4 v) noexcept { auto t = reduce_and_64(v); return static_cast<int32>(t + (t >> 32)); }
inline int16 reduce_and(int16x8 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t & (t >> 16)); }
inline int16 reduce_or (int16x8 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t | (t >> 16)); }
inline int16 reduce_xor(int16x8 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t ^ (t >> 16)); }
inline int16 reduce_add(int16x8 v) noexcept { auto t = reduce_and_32(v); return static_cast<int16>(t + (t >> 16)); }
inline int8  reduce_and(int8x16 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t & (t >> 8)); }
inline int8  reduce_or (int8x16 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t | (t >> 8)); }
inline int8  reduce_xor(int8x16 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t ^ (t >> 8)); }
inline int8  reduce_add(int8x16 v) noexcept { auto t = reduce_and_16(v); return static_cast<int8>(t + (t >> 8)); }


class int256
{
protected:
	__m256i reg{ 0 };
public:
	int256() = default;
	int256(__m256i v) noexcept : reg(v) {}
	explicit int256(std::nullptr_t) noexcept = delete;
	explicit int256(const int64* p) noexcept : reg(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))) {}
	explicit int256(const int32* p) noexcept : reg(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))) {}
	explicit int256(const int16* p) noexcept : reg(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))) {}
	explicit int256(const int8* p) noexcept : reg(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))) {}
	explicit int256(int64 v) noexcept : reg(_mm256_set1_epi64x(v)) {}
	explicit int256(int32 v) noexcept : reg(_mm256_set1_epi32(v)) {}
	explicit int256(int16 v) noexcept : reg(_mm256_set1_epi16(v)) {}
	explicit int256(int8 v) noexcept : reg(_mm256_set1_epi8(v)) {}
	int256(int64 e3, int64 e2, int64 e1, int64 e0) noexcept : reg(_mm256_set_epi64x(e3, e2, e1, e0)) {}
	int256(int32 e7, int32 e6, int32 e5, int32 e4, int32 e3, int32 e2, int32 e1, int32 e0) noexcept : reg(_mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0)) {}
	int256(int16 e15, int16 e14, int16 e13, int16 e12, int16 e11, int16 e10, int16 e9, int16 e8, 
		int16 e7, int16 e6, int16 e5, int16 e4, int16 e3, int16 e2, int16 e1, int16 e0) noexcept
		: reg(_mm256_set_epi16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)) {}
	int256(int8 e31, int8 e30, int8 e29, int8 e28, int8 e27, int8 e26, int8 e25, int8 e24,
		int8 e23, int8 e22, int8 e21, int8 e20, int8 e19, int8 e18, int8 e17, int8 e16,
		int8 e15, int8 e14, int8 e13, int8 e12, int8 e11, int8 e10, int8 e9, int8 e8,
		int8 e7, int8 e6, int8 e5, int8 e4, int8 e3, int8 e2, int8 e1, int8 e0) noexcept
		: reg(_mm256_set_epi8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16,
			e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)) {}

	operator __m256i() const noexcept { return reg; }

	template <uint index> int8   get_int8  () const noexcept { static_assert(index < 32); return _mm256_extract_epi8(reg, index); }
	template <uint index> int16  get_int16 () const noexcept { static_assert(index < 16); return _mm256_extract_epi16(reg, index); }
	template <uint index> int32  get_int32 () const noexcept { static_assert(index <  8); return _mm256_extract_epi32(reg, index); }
	template <uint index> int64  get_int64 () const noexcept { static_assert(index <  4); return _mm256_extract_epi64(reg, index); }
	template <uint index> int128 get_int128() const noexcept { static_assert(index <  2); return _mm256_extracti128_si256(reg, index); }

	template <uint index> void set_int8  (int8   v) noexcept { static_assert(index < 32); reg = _mm256_insert_epi8(reg, v, index); }
	template <uint index> void set_int16 (int16  v) noexcept { static_assert(index < 16); reg = _mm256_insert_epi16(reg, v, index); }
	template <uint index> void set_int32 (int32  v) noexcept { static_assert(index <  8); reg = _mm256_insert_epi32(reg, v, index); }
	template <uint index> void set_int64 (int64  v) noexcept { static_assert(index <  4); reg = _mm256_insert_epi64(reg, v, index); }
	template <uint index> void set_int128(int128 v) noexcept { static_assert(index <  2); reg = _mm256_inserti128_si256(reg, v, index); }

	template <uint index> int256 with_int8  (int8   v) noexcept { static_assert(index < 32); return _mm256_insert_epi8(reg, v, index); }
	template <uint index> int256 with_int16 (int16  v) noexcept { static_assert(index < 16); return _mm256_insert_epi16(reg, v, index); }
	template <uint index> int256 with_int32 (int32  v) noexcept { static_assert(index <  8); return _mm256_insert_epi32(reg, v, index); }
	template <uint index> int256 with_int64 (int64  v) noexcept { static_assert(index <  4); return _mm256_insert_epi64(reg, v, index); }
	template <uint index> int256 with_int128(int128 v) noexcept { static_assert(index <  2); return _mm256_inserti128_si256(reg, v, index); }

	bool operator==(int256 o) const noexcept { auto x = _mm256_xor_si256(reg, o.reg); return _mm256_testz_si256(x, x); }
	bool operator!=(int256 o) const noexcept { return !(*this == o); }

	int256 operator~() const noexcept { return andnot(reg, int256{ -1 }); }
	int256 operator&(int256 v) const noexcept { return _mm256_and_si256(reg, v); }
	int256 operator|(int256 v) const noexcept { return _mm256_or_si256(reg, v); }
	int256 operator^(int256 v) const noexcept { return _mm256_xor_si256(reg, v); }
	int256& operator&=(int256 v) noexcept { reg = *this & v; return *this; }
	int256& operator|=(int256 v) noexcept { reg = *this | v; return *this; }
	int256& operator^=(int256 v) noexcept { reg = *this ^ v; return *this; }
	friend int256 andnot(int256 l, int256 r) noexcept { return _mm256_andnot_si256(l, r); }
};

class int64x4 : public int256
{
public:
	int64x4() = default;
	int64x4(__m256i v) noexcept : int256(v) {}
	explicit int64x4(std::nullptr_t) noexcept = delete;
	explicit int64x4(const int64* p) noexcept : int256(p) {}
	explicit int64x4(int64 v) noexcept : int256(v) {}
	int64x4(int64 e3, int64 e2, int64 e1, int64 e0) noexcept : int256(e3, e2, e1, e0) {}

	template <uint index> int64 get() const noexcept { return get_int64<index>(); }
	template <uint index> void set(int64 v) noexcept { set_int64<index>(); }
	template <uint index> int256 with(int64 v) noexcept { return with_int64<index>(v); }

	int64x4 operator~() const noexcept { return andnot(reg, int64x4{ -1 }); }
	int64x4 operator-() const noexcept { return int64x4{} - *this; }
	int64x4 operator&(int64x4 v) const noexcept { return _mm256_and_si256(reg, v); }
	int64x4 operator|(int64x4 v) const noexcept { return _mm256_or_si256(reg, v); }
	int64x4 operator^(int64x4 v) const noexcept { return _mm256_xor_si256(reg, v); }
	int64x4 operator+(int64x4 v) const noexcept { return _mm256_add_epi64(reg, v.reg); }
	int64x4 operator-(int64x4 v) const noexcept { return _mm256_sub_epi64(reg, v.reg); }
	//int64x4 operator*(int64x4 v) const noexcept { return _mm256_mul_epi64(reg, v.reg); } // Instruction does not exist.
	// int64x4 operator/(int64x4 v) const noexcept { return _mm256_div_epi64(reg, v.reg); }
	// int64x4 operator%(int64x4 v) const noexcept { return _mm256_rem_epi64(reg, v.reg); }
	int64x4 operator<<(int64x4 v) const noexcept { return _mm256_sllv_epi64(reg, v.reg); }
	int64x4 operator>>(int64x4 v) const noexcept { return _mm256_srlv_epi64(reg, v.reg); }
	int64x4 operator<<(int v) const noexcept { return _mm256_slli_epi64(reg, v); }
	int64x4 operator>>(int v) const noexcept { return _mm256_srli_epi64(reg, v); }
	int64x4& operator&=(int64x4 v) noexcept { reg = *this & v; return *this; }
	int64x4& operator|=(int64x4 v) noexcept { reg = *this | v; return *this; }
	int64x4& operator^=(int64x4 v) noexcept { reg = *this ^ v; return *this; }
	friend int64x4 andnot(int64x4 l, int64x4 r) noexcept { return _mm256_andnot_si256(l, r); }

	int64x4& operator+=(int64x4 v) noexcept { reg = *this + v; return *this; }
	int64x4& operator-=(int64x4 v) noexcept { reg = *this - v; return *this; }
	//int64x4& operator*=(int64x4 v) noexcept { reg = *this * v; return *this; } // Instruction does not exist.
	// int64x4& operator/=(int64x4 v) noexcept { reg = *this / v; return *this; }
	int64x4& operator<<=(int64x4 v) noexcept { reg = *this << v; return *this; }
	int64x4& operator>>=(int64x4 v) noexcept { reg = *this >> v; return *this; }
	int64x4& operator<<=(int v) noexcept { reg = *this << v; return *this; }
	int64x4& operator>>=(int v) noexcept { reg = *this >> v; return *this; }
};

class int32x8 : public int256
{
public:
	int32x8() = default;
	int32x8(__m256i v) noexcept : int256(v) {}
	explicit int32x8(std::nullptr_t) noexcept = delete;
	explicit int32x8(const int32* p) noexcept : int256(p) {}
	explicit int32x8(int32 v) noexcept : int256(v) {}
	int32x8(int32 e7, int32 e6, int32 e5, int32 e4, int32 e3, int32 e2, int32 e1, int32 e0) noexcept : int256(e7, e6, e5, e4, e3, e2, e1, e0) {}

	template <uint index> int32 get() const noexcept { return get_int32<index>(); }
	template <uint index> void set(int32 v) noexcept { set_int32<index>(); }
	template <uint index> int256 with(int32 v) noexcept { return with_int32<index>(v); }

	int32x8 operator~() const noexcept { return andnot(reg, int32x8{ -1 }); }
	int32x8 operator-() const noexcept { return int32x8{} - *this; }
	int32x8 operator&(int32x8 v) const noexcept { return _mm256_and_si256(reg, v); }
	int32x8 operator|(int32x8 v) const noexcept { return _mm256_or_si256(reg, v); }
	int32x8 operator^(int32x8 v) const noexcept { return _mm256_xor_si256(reg, v); }
	int32x8 operator+(int32x8 v) const noexcept { return _mm256_add_epi32(reg, v.reg); }
	int32x8 operator-(int32x8 v) const noexcept { return _mm256_sub_epi32(reg, v.reg); }
	int32x8 operator*(int32x8 v) const noexcept { return _mm256_mul_epi32(reg, v.reg); }
	// int32x8 operator/(int32x8 v) const noexcept { return _mm256_div_epi32(reg, v.reg); }
	// int32x8 operator%(int32x8 v) const noexcept { return _mm256_rem_epi32(reg, v.reg); }
	int32x8 operator<<(int32x8 v) const noexcept { return _mm256_sllv_epi32(reg, v.reg); }
	int32x8 operator>>(int32x8 v) const noexcept { return _mm256_srlv_epi32(reg, v.reg); }
	int32x8 operator<<(int v) const noexcept { return _mm256_slli_epi32(reg, v); }
	int32x8 operator>>(int v) const noexcept { return _mm256_srli_epi32(reg, v); }
	int32x8& operator&=(int32x8 v) noexcept { reg = *this & v; return *this; }
	int32x8& operator|=(int32x8 v) noexcept { reg = *this | v; return *this; }
	int32x8& operator^=(int32x8 v) noexcept { reg = *this ^ v; return *this; }
	friend int32x8 andnot(int32x8 l, int32x8 r) noexcept { return _mm256_andnot_si256(l, r); }

	int32x8& operator+=(int32x8 v) noexcept { reg = *this + v; return *this; }
	int32x8& operator-=(int32x8 v) noexcept { reg = *this - v; return *this; }
	int32x8& operator*=(int32x8 v) noexcept { reg = *this * v; return *this; }
	// int32x8& operator/=(int32x8 v) noexcept { reg = *this / v; return *this; }
	int32x8& operator<<=(int32x8 v) noexcept { reg = *this << v; return *this; }
	int32x8& operator>>=(int32x8 v) noexcept { reg = *this >> v; return *this; }
	int32x8& operator<<=(int v) noexcept { reg = *this << v; return *this; }
	int32x8& operator>>=(int v) noexcept { reg = *this >> v; return *this; }
};

class int16x16 : public int256
{
public:
	int16x16() = default;
	int16x16(__m256i v) noexcept : int256(v) {}
	explicit int16x16(std::nullptr_t) noexcept = delete;
	explicit int16x16(const int16* p) noexcept : int256(p) {}
	explicit int16x16(int16 v) noexcept : int256(v) {}
	int16x16(int16 e15, int16 e14, int16 e13, int16 e12, int16 e11, int16 e10, int16 e9, int16 e16,
		int16 e7, int16 e6, int16 e5, int16 e4, int16 e3, int16 e2, int16 e1, int16 e0) noexcept
		: int256(e15, e14, e13, e12, e11, e10, e9, e16, e7, e6, e5, e4, e3, e2, e1, e0) {}

	template <uint index> int16 get() const noexcept { return get_int16<index>(); }
	template <uint index> void set(int16 v) noexcept { set_int16<index>(); }
	template <uint index> int256 with(int16 v) noexcept { return with_int16<index>(v); }

	int16x16 operator~() const noexcept { return andnot(reg, int16x16{ -1 }); }
	int16x16 operator-() const noexcept { return int16x16{} - *this; }
	int16x16 operator&(int16x16 v) const noexcept { return _mm256_and_si256(reg, v); }
	int16x16 operator|(int16x16 v) const noexcept { return _mm256_or_si256(reg, v); }
	int16x16 operator^(int16x16 v) const noexcept { return _mm256_xor_si256(reg, v); }
	int16x16 operator+(int16x16 v) const noexcept { return _mm256_add_epi16(reg, v.reg); }
	int16x16 operator-(int16x16 v) const noexcept { return _mm256_sub_epi16(reg, v.reg); }
	//int16x16 operator*(int16x16 v) const noexcept { return _mm256_mul_epi16(reg, v.reg); } // Instruction does not exist.
	// int16x16 operator/(int16x16 v) const noexcept { return _mm256_div_epi16(reg, v.reg); }
	// int16x16 operator%(int16x16 v) const noexcept { return _mm256_rem_epi16(reg, v.reg); }
	//int16x16 operator<<(int16x16 v) const noexcept { return _mm256_sllv_epi16(reg, v.reg); } // Instruction requires AVX512VL and AVX512BW.
	//int16x16 operator>>(int16x16 v) const noexcept { return _mm256_srlv_epi16(reg, v.reg); } // Instruction requires AVX512VL and AVX512BW.
	int16x16 operator<<(int v) const noexcept { return _mm256_slli_epi16(reg, v); }
	int16x16 operator>>(int v) const noexcept { return _mm256_srli_epi16(reg, v); }
	int16x16& operator&=(int16x16 v) noexcept { reg = *this & v; return *this; }
	int16x16& operator|=(int16x16 v) noexcept { reg = *this | v; return *this; }
	int16x16& operator^=(int16x16 v) noexcept { reg = *this ^ v; return *this; }
	friend int16x16 andnot(int16x16 l, int16x16 r) noexcept { return _mm256_andnot_si256(l, r); }

	int16x16& operator+=(int16x16 v) noexcept { reg = *this + v; return *this; }
	int16x16& operator-=(int16x16 v) noexcept { reg = *this - v; return *this; }
	//int16x16& operator*=(int16x16 v) noexcept { reg = *this * v; return *this; } // Instruction does not exist.
	// int16x16& operator/=(int16x16 v) noexcept { reg = *this / v; return *this; }
	//int16x16& operator<<=(int16x16 v) noexcept { reg = *this << v; return *this; } // Instruction requires AVX512VL and AVX512BW.
	//int16x16& operator>>=(int16x16 v) noexcept { reg = *this >> v; return *this; } // Instruction requires AVX512VL and AVX512BW.
	int16x16& operator<<=(int v) noexcept { reg = *this << v; return *this; }
	int16x16& operator>>=(int v) noexcept { reg = *this >> v; return *this; }
};

class int8x32 : public int256
{
public:
	int8x32() = default;
	int8x32(__m256i v) noexcept : int256(v) {}
	explicit int8x32(std::nullptr_t) noexcept = delete;
	explicit int8x32(const int8* p) noexcept : int256(p) {}
	explicit int8x32(int8 v) noexcept : int256(v) {}
	int8x32(int8 e31, int8 e30, int8 e29, int8 e28, int8 e27, int8 e26, int8 e25, int8 e24,
		int8 e23, int8 e22, int8 e21, int8 e20, int8 e19, int8 e18, int8 e17, int8 e16,
		int8 e15, int8 e14, int8 e13, int8 e12, int8 e11, int8 e10, int8 e9, int8 e8,
		int8 e7, int8 e6, int8 e5, int8 e4, int8 e3, int8 e2, int8 e1, int8 e0) noexcept
		: int256(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16,
			e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) {}

	template <uint index> int8 get() const noexcept { return get_int8<index>(); }
	template <uint index> void set(int8 v) noexcept { set_int8<index>(); }
	template <uint index> int256 with(int8 v) noexcept { return with_int8<index>(v); }

	int8x32 operator~() const noexcept { return andnot(reg, int8x32{ -1 }); }
	int8x32 operator-() const noexcept { return int8x32{} - *this; }
	int8x32 operator&(int8x32 v) const noexcept { return _mm256_and_si256(reg, v); }
	int8x32 operator|(int8x32 v) const noexcept { return _mm256_or_si256(reg, v); }
	int8x32 operator^(int8x32 v) const noexcept { return _mm256_xor_si256(reg, v); }
	int8x32 operator+(int8x32 v) const noexcept { return _mm256_add_epi8(reg, v.reg); }
	int8x32 operator-(int8x32 v) const noexcept { return _mm256_sub_epi8(reg, v.reg); }
	//int8x32 operator*(int8x32 v) const noexcept { return _mm256_mul_epi8(reg, v.reg); } // Instruction does not exist.
	// int8x32 operator/(int8x32 v) const noexcept { return _mm256_div_epi8(reg, v.reg); }
	// int8x32 operator%(int8x32 v) const noexcept { return _mm256_rem_epi8(reg, v.reg); }
	//int8x32 operator<<(int8x32 v) const noexcept { return _mm256_sllv_epi8(reg, v.reg); } // Instruction does not exist.
	//int8x32 operator>>(int8x32 v) const noexcept { return _mm256_srlv_epi8(reg, v.reg); } // Instruction does not exist.
	//int8x32 operator<<(int v) const noexcept { return _mm256_slli_epi8(reg, v); } // Instruction does not exist.
	//int8x32 operator>>(int v) const noexcept { return _mm256_srli_epi8(reg, v); } // Instruction does not exist.
	int8x32& operator&=(int8x32 v) noexcept { reg = *this & v; return *this; }
	int8x32& operator|=(int8x32 v) noexcept { reg = *this | v; return *this; }
	int8x32& operator^=(int8x32 v) noexcept { reg = *this ^ v; return *this; }
	friend int8x32 andnot(int8x32 l, int8x32 r) noexcept { return _mm256_andnot_si256(l, r); }

	int8x32& operator+=(int8x32 v) noexcept { reg = *this + v; return *this; }
	int8x32& operator-=(int8x32 v) noexcept { reg = *this - v; return *this; }
	//int8x32& operator*=(int8x32 v) noexcept { reg = *this * v; return *this; } // Instruction does not exist.
	// int8x32& operator/=(int8x32 v) noexcept { reg = *this / v; return *this; }
	//int8x32& operator<<=(int8x32 v) noexcept { reg = *this << v; return *this; } // Instruction does not exist.
	//int8x32& operator>>=(int8x32 v) noexcept { reg = *this >> v; return *this; } // Instruction does not exist.
	//int8x32& operator<<=(int v) noexcept { reg = *this << v; return *this; } // Instruction does not exist.
	//int8x32& operator>>=(int v) noexcept { reg = *this >> v; return *this; } // Instruction does not exist.
};


// unpackhi({a,_,c,_}, {e,_,g,_}) -> {e,a,g,c}
inline int256 unpackhi_64(int256 x, int256 y) noexcept { return _mm256_unpackhi_epi64(x, y); }
inline int64x4 unpackhi(int64x4 x, int64x4 y) noexcept { return _mm256_unpackhi_epi64(x, y); }

// unpacklo({_,b,_,d}, {_,f,_,h}) -> {f,b,h,d}
inline int256 unpacklo_64(int256 x, int256 y) noexcept { return _mm256_unpacklo_epi64(x, y); }
inline int64x4 unpacklo(int64x4 x, int64x4 y) noexcept { return _mm256_unpacklo_epi64(x, y); }

// unpackhi({a,b,_,_,e,f,_,_}, {i,j,_,_,m,n,_,_}) -> {i,a,j,b,m,e,n,f}
inline int256 unpackhi_32(int256 x, int256 y) noexcept { return _mm256_unpackhi_epi32(x, y); }
inline int32x8 unpackhi(int32x8 x, int32x8 y) noexcept { return _mm256_unpackhi_epi32(x, y); }

// unpacklo({_,_,c,d,_,_,g,h}, {_,_,k,l,_,_,o,p})) -> {k,c,l,d,o,g,p,h}
inline int256 unpacklo_32(int256 x, int256 y) noexcept { return _mm256_unpacklo_epi32(x, y); }
inline int32x8 unpacklo(int32x8 x, int32x8 y) noexcept { return _mm256_unpacklo_epi32(x, y); }


// unpackhi({a,b,c,d,_,_,_,_,i,j,k,l,_,_,_,_}, {a2,b2,c2,d2,_,_,_,_,i2,j2,k2,l2,_,_,_,_}) -> {a2,a,b2,b,c2,c,d2,d,i2,i,j2,j,k2,k,l2,l}
inline int256 unpackhi_16(int256 x, int256 y) noexcept { return _mm256_unpackhi_epi16(x, y); }
inline int16x16 unpackhi(int16x16 x, int16x16 y) noexcept { return _mm256_unpackhi_epi16(x, y); }

// unpacklo({_,_,_,_,e,f,g,h,_,_,_,_,m,n,o,p}, {_,_,_,_,e2,f2,g2,h2,_,_,_,_,m2,n2,o2,p2})) -> {e2,e,f2,f,g2,g,h2,h,m2,m,n2,n,o2,o,p2,p}
inline int256 unpacklo_16(int256 x, int256 y) noexcept { return _mm256_unpacklo_epi16(x, y); }
inline int16x16 unpacklo(int16x16 x, int16x16 y) noexcept { return _mm256_unpacklo_epi16(x, y); }

// unpackhi({a1,b1,c1,d1,e1,f1,g1,h1,_,_,_,_,_,_,_,_,i1,j1,k1,l1,m1,n1,o1,p1,_,_,_,_,_,_,_,_},{a2,b2,c2,d2,e2,f2,g2,h2,_,_,_,_,_,_,_,_,i2,j2,k2,l2,m2,n2,o2,p2,_,_,_,_,_,_,_,_})
//   -> {a2,a1,b2,b1,c2,c1,d2,d1,e2,e1,f2,f1,g2,g1,h2,h1,i2,i1,j2,j1,k2,k1,l2,l1,m2,m1,n2,n1,o2,o1,p2,p1}
inline int256 unpackhi_8(int256 x, int256 y) noexcept { return _mm256_unpackhi_epi8(x, y); }
inline int8x32 unpackhi(int8x32 x, int8x32 y) noexcept { return _mm256_unpackhi_epi8(x, y); }

// unpacklo({_,_,_,_,_,_,_,_,a1,b1,c1,d1,e1,f1,g1,h1,_,_,_,_,_,_,_,_,i1,j1,k1,l1,m1,n1,o1,p1},{_,_,_,_,_,_,_,_,a2,b2,c2,d2,e2,f2,g2,h2,_,_,_,_,_,_,_,_,i2,j2,k2,l2,m2,n2,o2,p2})
//   -> {a2,a1,b2,b1,c2,c1,d2,d1,e2,e1,f2,f1,g2,g1,h2,h1,i2,i1,j2,j1,k2,k1,l2,l1,m2,m1,n2,n1,o2,o1,p2,p1}
inline int256 unpacklo_8(int256 x, int256 y) noexcept { return _mm256_unpacklo_epi8(x, y); }
inline int8x32 unpacklo(int8x32 x, int8x32 y) noexcept { return _mm256_unpacklo_epi8(x, y); }

// returns a==b ? -1 : 0
inline int256 cmpeq_64(int256 a, int256 b) noexcept { return _mm256_cmpeq_epi64(a, b); }
inline int256 cmpeq_32(int256 a, int256 b) noexcept { return _mm256_cmpeq_epi32(a, b); }
inline int256 cmpeq_16(int256 a, int256 b) noexcept { return _mm256_cmpeq_epi16(a, b); }
inline int256 cmpeq_8 (int256 a, int256 b) noexcept { return _mm256_cmpeq_epi8 (a, b); }
inline int64x4  cmpeq(int64x4  a, int64x4  b) noexcept { return _mm256_cmpeq_epi64(a, b); }
inline int32x8  cmpeq(int32x8  a, int32x8  b) noexcept { return _mm256_cmpeq_epi32(a, b); }
inline int16x16 cmpeq(int16x16 a, int16x16 b) noexcept { return _mm256_cmpeq_epi16(a, b); }
inline int8x32  cmpeq(int8x32  a, int8x32  b) noexcept { return _mm256_cmpeq_epi8 (a, b); }

// returns a>b ? -1 : 0
inline int256 cmpgt_64(int256 a, int256 b) noexcept { return _mm256_cmpgt_epi64(a, b); }
inline int256 cmpgt_32(int256 a, int256 b) noexcept { return _mm256_cmpgt_epi32(a, b); }
inline int256 cmpgt_16(int256 a, int256 b) noexcept { return _mm256_cmpgt_epi16(a, b); }
inline int256 cmpgt_8 (int256 a, int256 b) noexcept { return _mm256_cmpgt_epi8 (a, b); }
inline int64x4  cmpgt(int64x4  a, int64x4  b) noexcept { return _mm256_cmpgt_epi64(a, b); }
inline int32x8  cmpgt(int32x8  a, int32x8  b) noexcept { return _mm256_cmpgt_epi32(a, b); }
inline int16x16 cmpgt(int16x16 a, int16x16 b) noexcept { return _mm256_cmpgt_epi16(a, b); }
inline int8x32  cmpgt(int8x32  a, int8x32  b) noexcept { return _mm256_cmpgt_epi8 (a, b); }

inline int256 abs_64(int256 v) noexcept { return _mm256_abs_epi64(v); }
inline int256 abs_32(int256 v) noexcept { return _mm256_abs_epi32(v); }
inline int256 abs_16(int256 v) noexcept { return _mm256_abs_epi16(v); }
inline int256 abs_8 (int256 v) noexcept { return _mm256_abs_epi8 (v); }
inline int64x4  abs(int64x4  v) noexcept { return _mm256_abs_epi64(v); }
inline int32x8  abs(int32x8  v) noexcept { return _mm256_abs_epi32(v); }
inline int16x16 abs(int16x16 v) noexcept { return _mm256_abs_epi16(v); }
inline int8x32  abs(int8x32  v) noexcept { return _mm256_abs_epi8 (v); }

inline int64 reduce_and_64(int256 v) noexcept { return reduce_and_64(v.get_int128<0>() & v.get_int128<1>()); }
inline int64 reduce_or_64(int256 v) noexcept { return reduce_or_64(v.get_int128<0>() | v.get_int128<1>()); }
inline int64 reduce_xor_64(int256 v) noexcept { return reduce_xor_64(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int64 reduce_add_64(int256 v) noexcept { return reduce_add_64(v.get_int128<0>()) + reduce_add_64(v.get_int128<1>()); }
inline int32 reduce_and_32(int256 v) noexcept { return reduce_and_32(v.get_int128<0>() & v.get_int128<1>()); }
inline int32 reduce_or_32(int256 v) noexcept { return reduce_or_32(v.get_int128<0>() | v.get_int128<1>()); }
inline int32 reduce_xor_32(int256 v) noexcept { return reduce_xor_32(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int32 reduce_add_32(int256 v) noexcept { return reduce_add_32(v.get_int128<0>()) + reduce_add_32(v.get_int128<1>()); }
inline int16 reduce_and_16(int256 v) noexcept { return reduce_and_16(v.get_int128<0>() & v.get_int128<1>()); }
inline int16 reduce_or_16(int256 v) noexcept { return reduce_or_16(v.get_int128<0>() | v.get_int128<1>()); }
inline int16 reduce_xor_16(int256 v) noexcept { return reduce_xor_16(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int16 reduce_add_16(int256 v) noexcept { return reduce_add_16(v.get_int128<0>()) + reduce_add_16(v.get_int128<1>()); }
inline int8  reduce_and_8(int256 v) noexcept { return reduce_and_8(v.get_int128<0>() & v.get_int128<1>()); }
inline int8  reduce_or_8(int256 v) noexcept { return reduce_or_8(v.get_int128<0>() | v.get_int128<1>()); }
inline int8  reduce_xor_8(int256 v) noexcept { return reduce_xor_8(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int8  reduce_add_8(int256 v) noexcept { return reduce_add_8(v.get_int128<0>()) + reduce_add_8(v.get_int128<1>()); }

inline int64 reduce_and(int64x4 v) noexcept { return reduce_and_64(v.get_int128<0>() & v.get_int128<1>()); }
inline int64 reduce_or(int64x4 v) noexcept { return reduce_or_64(v.get_int128<0>() | v.get_int128<1>()); }
inline int64 reduce_xor(int64x4 v) noexcept { return reduce_xor_64(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int64 reduce_add(int64x4 v) noexcept { return reduce_add_64(v.get_int128<0>()) + reduce_add_64(v.get_int128<1>()); }
inline int32 reduce_and(int32x8 v) noexcept { return reduce_and_32(v.get_int128<0>() & v.get_int128<1>()); }
inline int32 reduce_or(int32x8 v) noexcept { return reduce_or_32(v.get_int128<0>() | v.get_int128<1>()); }
inline int32 reduce_xor(int32x8 v) noexcept { return reduce_xor_32(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int32 reduce_add(int32x8 v) noexcept { return reduce_add_32(v.get_int128<0>()) + reduce_add_32(v.get_int128<1>()); }
inline int16 reduce_and(int16x16 v) noexcept { return reduce_and_16(v.get_int128<0>() & v.get_int128<1>()); }
inline int16 reduce_or(int16x16 v) noexcept { return reduce_or_16(v.get_int128<0>() | v.get_int128<1>()); }
inline int16 reduce_xor(int16x16 v) noexcept { return reduce_xor_16(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int16 reduce_add(int16x16 v) noexcept { return reduce_add_16(v.get_int128<0>()) + reduce_add_16(v.get_int128<1>()); }
inline int8  reduce_and(int8x32 v) noexcept { return reduce_and_8(v.get_int128<0>() & v.get_int128<1>()); }
inline int8  reduce_or(int8x32 v) noexcept { return reduce_or_8(v.get_int128<0>() | v.get_int128<1>()); }
inline int8  reduce_xor(int8x32 v) noexcept { return reduce_xor_8(v.get_int128<0>() ^ v.get_int128<1>()); }
inline int8  reduce_add(int8x32 v) noexcept { return reduce_add_8(v.get_int128<0>()) + reduce_add_8(v.get_int128<1>()); }

#endif
