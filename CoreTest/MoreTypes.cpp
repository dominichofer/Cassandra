#include "pch.h"
#include "Core/Core.h"
#include <array>

namespace MoreTypes_128
{
	TEST(int128, constructor_equivalence)
	{
		__m128i a = _mm_set_epi64x(0x0102030405060708LL, 0x090A0B0C0D0E0F10LL);
		__m128i b = _mm_set_epi32(0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10);
		__m128i c = _mm_set_epi16(0x0102, 0x0304, 0x0506, 0x0708, 0x090A, 0x0B0C, 0x0D0E, 0x0F10);
		__m128i d = _mm_set_epi8(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10);

		ASSERT_EQ(int128(a), int128(b));
		ASSERT_EQ(int128(a), int128(c));
		ASSERT_EQ(int128(a), int128(d));

		int128 e{ 0x0102030405060708LL, 0x090A0B0C0D0E0F10LL };
		int128 f{ 0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10 };
		int128 g{ 0x0102, 0x0304, 0x0506, 0x0708, 0x090A, 0x0B0C, 0x0D0E, 0x0F10 };
		int128 h{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10 };

		ASSERT_EQ(a, e);
		ASSERT_EQ(a, f);
		ASSERT_EQ(a, g);
		ASSERT_EQ(a, h);

		const std::array<int64, 2> arr_i{ 0x090A0B0C0D0E0F10LL, 0x0102030405060708LL };
		const std::array<int32, 4> arr_j{ 0x0D0E0F10, 0x090A0B0C, 0x05060708, 0x01020304 };
		const std::array<int16, 8> arr_k{ 0x0F10, 0x0D0E, 0x0B0C, 0x090A, 0x0708, 0x0506, 0x0304, 0x0102 };
		const std::array<int8, 16> arr_l{ 0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01 };

		int128 i{ arr_i.data() };
		int128 j{ arr_j.data() };
		int128 k{ arr_k.data() };
		int128 l{ arr_l.data() };

		ASSERT_EQ(a, i);
		ASSERT_EQ(a, j);
		ASSERT_EQ(a, k);
		ASSERT_EQ(a, l);
	}

	TEST(int128, broadcast)
	{
		int64 a = 0x0102030405060708LL; // arbitrary
		int32 b = 0x01020304; // arbitrary
		int16 c = 0x0102; // arbitrary
		int8 d = 0x12; // arbitrary

		ASSERT_EQ(
			int128(a, a),
			int128(a)
		);
		ASSERT_EQ(
			int128(b, b, b, b),
			int128(b)
		);
		ASSERT_EQ(
			int128(c, c, c, c, c, c, c, c),
			int128(c)
		);
		ASSERT_EQ(
			int128(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d),
			int128(d)
		);
	}

	TEST(int128, compare)
	{
		ASSERT_TRUE(int128(1) == int128(1));
		ASSERT_FALSE(int128(1) == int128(2));

		ASSERT_TRUE(int128(1) != int128(2));
		ASSERT_FALSE(int128(1) != int128(1));
	}

	TEST(int128, insert)
	{
		int64 a = 0x0102030405060708LL; // arbitrary
		int32 b = 0x01020304; // arbitrary
		int16 c = 0x0102; // arbitrary
		int8 d = 0x12; // arbitrary

		ASSERT_EQ(int128(0, a), int128().with_int64<0>(a));
		ASSERT_EQ(int128(a, 0), int128().with_int64<1>(a));

		ASSERT_EQ(int128(0, 0, 0, b), int128().with_int32<0>(b));
		ASSERT_EQ(int128(0, 0, b, 0), int128().with_int32<1>(b));
		ASSERT_EQ(int128(0, b, 0, 0), int128().with_int32<2>(b));
		ASSERT_EQ(int128(b, 0, 0, 0), int128().with_int32<3>(b));

		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, c), int128().with_int16<0>(c));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, c, 0), int128().with_int16<1>(c));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, c, 0, 0), int128().with_int16<2>(c));
		ASSERT_EQ(int128(0, 0, 0, 0, c, 0, 0, 0), int128().with_int16<3>(c));
		ASSERT_EQ(int128(0, 0, 0, c, 0, 0, 0, 0), int128().with_int16<4>(c));
		ASSERT_EQ(int128(0, 0, c, 0, 0, 0, 0, 0), int128().with_int16<5>(c));
		ASSERT_EQ(int128(0, c, 0, 0, 0, 0, 0, 0), int128().with_int16<6>(c));
		ASSERT_EQ(int128(c, 0, 0, 0, 0, 0, 0, 0), int128().with_int16<7>(c));

		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d), int128().with_int8< 0>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0), int128().with_int8< 1>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0), int128().with_int8< 2>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0), int128().with_int8< 3>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0), int128().with_int8< 4>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0), int128().with_int8< 5>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0), int128().with_int8< 6>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0), int128().with_int8< 7>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8< 8>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8< 9>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8<10>(d));
		ASSERT_EQ(int128(0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8<11>(d));
		ASSERT_EQ(int128(0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8<12>(d));
		ASSERT_EQ(int128(0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8<13>(d));
		ASSERT_EQ(int128(0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8<14>(d));
		ASSERT_EQ(int128(d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int128().with_int8<15>(d));
	}

	TEST(int128, extract)
	{
		int64 a = 0x0102030405060708LL; // arbitrary
		int32 b = 0x01020304; // arbitrary
		int16 c = 0x0102; // arbitrary
		int8 d = 0x12; // arbitrary

		ASSERT_EQ(int128(0, a).get_int64<0>(), a);
		ASSERT_EQ(int128(a, 0).get_int64<1>(), a);

		ASSERT_EQ(int128(0, 0, 0, b).get_int32<0>(), b);
		ASSERT_EQ(int128(0, 0, b, 0).get_int32<1>(), b);
		ASSERT_EQ(int128(0, b, 0, 0).get_int32<2>(), b);
		ASSERT_EQ(int128(b, 0, 0, 0).get_int32<3>(), b);

		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, c).get_int16<0>(), c);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, c, 0).get_int16<1>(), c);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, c, 0, 0).get_int16<2>(), c);
		ASSERT_EQ(int128(0, 0, 0, 0, c, 0, 0, 0).get_int16<3>(), c);
		ASSERT_EQ(int128(0, 0, 0, c, 0, 0, 0, 0).get_int16<4>(), c);
		ASSERT_EQ(int128(0, 0, c, 0, 0, 0, 0, 0).get_int16<5>(), c);
		ASSERT_EQ(int128(0, c, 0, 0, 0, 0, 0, 0).get_int16<6>(), c);
		ASSERT_EQ(int128(c, 0, 0, 0, 0, 0, 0, 0).get_int16<7>(), c);

		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d).get_int8< 0>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0).get_int8< 1>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0).get_int8< 2>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0).get_int8< 3>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0).get_int8< 4>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0).get_int8< 5>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0).get_int8< 6>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0).get_int8< 7>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0).get_int8< 8>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8< 9>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<10>(), d);
		ASSERT_EQ(int128(0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<11>(), d);
		ASSERT_EQ(int128(0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<12>(), d);
		ASSERT_EQ(int128(0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<13>(), d);
		ASSERT_EQ(int128(0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<14>(), d);
		ASSERT_EQ(int128(d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<15>(), d);
	}

	TEST(int128, bitwise_logical)
	{
		int64 a = 0x0102030405060708LL; // arbitrary
		int64 b = 0x3040506070801020LL; // arbitrary

		ASSERT_EQ(~int128(a), int128(~a));
		ASSERT_EQ(int128(a) & int128(b), int128(a & b));
		ASSERT_EQ(int128(a) | int128(b), int128(a | b));
		ASSERT_EQ(int128(a) ^ int128(b), int128(a ^ b));
		ASSERT_EQ(andnot(int128(a), int128(b)), int128(~a & b));
	}
}

namespace MoreTypes_256
{
	TEST(int256, constructor_equivalence)
	{
		__m256i a = _mm256_set_epi64x(0x0102030405060708LL, 0x090A0B0C0D0E0F10LL, 0x1112131415161718LL, 0x191A1B1C1D1E1F20LL);
		__m256i b = _mm256_set_epi32(0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10, 0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20);
		__m256i c = _mm256_set_epi16(0x0102, 0x0304, 0x0506, 0x0708, 0x090A, 0x0B0C, 0x0D0E, 0x0F10, 0x1112, 0x1314, 0x1516, 0x1718, 0x191A, 0x1B1C, 0x1D1E, 0x1F20);
		__m256i d = _mm256_set_epi8(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20);

		ASSERT_EQ(int256(a), int256(b));
		ASSERT_EQ(int256(a), int256(c));
		ASSERT_EQ(int256(a), int256(d));

		int256 e{ 0x0102030405060708LL, 0x090A0B0C0D0E0F10LL, 0x1112131415161718LL, 0x191A1B1C1D1E1F20LL };
		int256 f{ 0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10, 0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20 };
		int256 g{ 0x0102, 0x0304, 0x0506, 0x0708, 0x090A, 0x0B0C, 0x0D0E, 0x0F10, 0x1112, 0x1314, 0x1516, 0x1718, 0x191A, 0x1B1C, 0x1D1E, 0x1F20 };
		int256 h{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20 };

		ASSERT_EQ(a, e);
		ASSERT_EQ(a, f);
		ASSERT_EQ(a, g);
		ASSERT_EQ(a, h);

		const std::array<int64,  4> arr_i{ 0x191A1B1C1D1E1F20LL, 0x1112131415161718LL, 0x90A0B0C0D0E0F10LL, 0x102030405060708LL };
		const std::array<int32,  8> arr_j{ 0x1D1E1F20, 0x191A1B1C, 0x15161718, 0x11121314, 0xD0E0F10, 0x90A0B0C, 0x5060708, 0x1020304 };
		const std::array<int16, 16> arr_k{ 0x1F20, 0x1D1E, 0x1B1C, 0x191A, 0x1718, 0x1516, 0x1314, 0x1112, 0xF10, 0xD0E, 0xB0C, 0x90A, 0x708, 0x506, 0x304, 0x102 };
		const std::array<int8,  32> arr_l{ 0x20, 0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01 };

		int256 i{ arr_i.data() };
		int256 j{ arr_j.data() };
		int256 k{ arr_k.data() };
		int256 l{ arr_l.data() };

		ASSERT_EQ(a, i);
		ASSERT_EQ(a, j);
		ASSERT_EQ(a, k);
		ASSERT_EQ(a, l);
	}

	TEST(int256, broadcast)
	{
		int64 a = 0x0102030405060708LL; // arbitrary
		int32 b = 0x01020304; // arbitrary
		int16 c = 0x0102; // arbitrary
		int8 d = 0x12; // arbitrary

		ASSERT_EQ(
			int256(a, a, a, a),
			int256(a)
		);
		ASSERT_EQ(
			int256(b, b, b, b, b, b, b, b),
			int256(b)
		);
		ASSERT_EQ(
			int256(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c),
			int256(c)
		);
		ASSERT_EQ(
			int256(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d),
			int256(d)
		);
	}

	TEST(int256, compare)
	{
		ASSERT_TRUE(int256(1) == int256(1));
		ASSERT_FALSE(int256(1) == int256(2));

		ASSERT_TRUE(int256(1) != int256(2));
		ASSERT_FALSE(int256(1) != int256(1));
	}

	TEST(int256, insert)
	{
		int64 b = 0x0102030405060708LL; // arbitrary
		int32 c = 0x01020304; // arbitrary
		int16 d = 0x0102; // arbitrary
		int8 e = 0x12; // arbitrary

		ASSERT_EQ(int256(0, 0, 0, b), int256().with_int64<0>(b));
		ASSERT_EQ(int256(0, 0, b, 0), int256().with_int64<1>(b));
		ASSERT_EQ(int256(0, b, 0, 0), int256().with_int64<2>(b));
		ASSERT_EQ(int256(b, 0, 0, 0), int256().with_int64<3>(b));

		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, c), int256().with_int32<0>(c));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, c, 0), int256().with_int32<1>(c));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, c, 0, 0), int256().with_int32<2>(c));
		ASSERT_EQ(int256(0, 0, 0, 0, c, 0, 0, 0), int256().with_int32<3>(c));
		ASSERT_EQ(int256(0, 0, 0, c, 0, 0, 0, 0), int256().with_int32<4>(c));
		ASSERT_EQ(int256(0, 0, c, 0, 0, 0, 0, 0), int256().with_int32<5>(c));
		ASSERT_EQ(int256(0, c, 0, 0, 0, 0, 0, 0), int256().with_int32<6>(c));
		ASSERT_EQ(int256(c, 0, 0, 0, 0, 0, 0, 0), int256().with_int32<7>(c));

		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d), int256().with_int16< 0>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0), int256().with_int16< 1>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0), int256().with_int16< 2>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0), int256().with_int16< 3>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0), int256().with_int16< 4>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0), int256().with_int16< 5>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0), int256().with_int16< 6>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0), int256().with_int16< 7>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16< 8>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16< 9>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16<10>(d));
		ASSERT_EQ(int256(0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16<11>(d));
		ASSERT_EQ(int256(0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16<12>(d));
		ASSERT_EQ(int256(0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16<13>(d));
		ASSERT_EQ(int256(0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16<14>(d));
		ASSERT_EQ(int256(d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int16<15>(d));

		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e), int256().with_int8< 0>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0), int256().with_int8< 1>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0), int256().with_int8< 2>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0), int256().with_int8< 3>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0), int256().with_int8< 4>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0), int256().with_int8< 5>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0), int256().with_int8< 6>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0), int256().with_int8< 7>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8< 8>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8< 9>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<10>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<11>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<12>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<13>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<14>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<15>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<16>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<17>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<18>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<19>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<20>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<21>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<22>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<23>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<24>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<25>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<26>(e));
		ASSERT_EQ(int256(0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<27>(e));
		ASSERT_EQ(int256(0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<28>(e));
		ASSERT_EQ(int256(0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<29>(e));
		ASSERT_EQ(int256(0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<30>(e));
		ASSERT_EQ(int256(e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), int256().with_int8<31>(e));
	}

	TEST(int256, extract)
	{
		int64 b = 0x0102030405060708LL; // arbitrary
		int32 c = 0x01020304; // arbitrary
		int16 d = 0x0102; // arbitrary
		int8 e = 0x12; // arbitrary

		ASSERT_EQ(int256(0, 0, 0, b).get_int64<0>(), b);
		ASSERT_EQ(int256(0, 0, b, 0).get_int64<1>(), b);
		ASSERT_EQ(int256(0, b, 0, 0).get_int64<2>(), b);
		ASSERT_EQ(int256(b, 0, 0, 0).get_int64<3>(), b);

		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, c).get_int32<0>(), c);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, c, 0).get_int32<1>(), c);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, c, 0, 0).get_int32<2>(), c);
		ASSERT_EQ(int256(0, 0, 0, 0, c, 0, 0, 0).get_int32<3>(), c);
		ASSERT_EQ(int256(0, 0, 0, c, 0, 0, 0, 0).get_int32<4>(), c);
		ASSERT_EQ(int256(0, 0, c, 0, 0, 0, 0, 0).get_int32<5>(), c);
		ASSERT_EQ(int256(0, c, 0, 0, 0, 0, 0, 0).get_int32<6>(), c);
		ASSERT_EQ(int256(c, 0, 0, 0, 0, 0, 0, 0).get_int32<7>(), c);

		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d).get_int16< 0>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0).get_int16< 1>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0).get_int16< 2>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0).get_int16< 3>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0).get_int16< 4>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0).get_int16< 5>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0).get_int16< 6>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0).get_int16< 7>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0).get_int16< 8>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int16< 9>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int16<10>(), d);
		ASSERT_EQ(int256(0, 0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int16<11>(), d);
		ASSERT_EQ(int256(0, 0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int16<12>(), d);
		ASSERT_EQ(int256(0, 0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int16<13>(), d);
		ASSERT_EQ(int256(0, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int16<14>(), d);
		ASSERT_EQ(int256(d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int16<15>(), d);

		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e).get_int8< 0>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0).get_int8< 1>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0).get_int8< 2>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0).get_int8< 3>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0).get_int8< 4>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0).get_int8< 5>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0).get_int8< 6>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0).get_int8< 7>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0).get_int8< 8>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8< 9>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<10>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<11>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<12>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<13>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<14>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<15>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<16>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<17>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<18>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<19>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<20>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<21>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<22>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<23>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<24>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<25>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<26>(), e);
		ASSERT_EQ(int256(0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<27>(), e);
		ASSERT_EQ(int256(0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<28>(), e);
		ASSERT_EQ(int256(0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<29>(), e);
		ASSERT_EQ(int256(0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<30>(), e);
		ASSERT_EQ(int256(e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).get_int8<31>(), e);
	}

	TEST(int256, bitwise_logical)
	{
		int64 a = 0x0102030405060708LL; // arbitrary
		int64 b = 0x3040506070801020LL; // arbitrary

		ASSERT_EQ(~int256(a), int256(~a));
		ASSERT_EQ(int256(a) & int256(b), int256(a & b));
		ASSERT_EQ(int256(a) | int256(b), int256(a | b));
		ASSERT_EQ(int256(a) ^ int256(b), int256(a ^ b));
		ASSERT_EQ(andnot(int256(a), int256(b)), int256(~a & b));
	}
}