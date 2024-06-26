#include "Base/Base.h"
#include "Helpers.h"
#include <array>

class SumPow3Cache
{
	std::array<uint32_t, (1ULL << 16)> cache{};

	static uint32_t sum_pow3(uint64_t exp)
	{
		uint32_t sum = 0;
		while (exp)
		{
			sum += pown(3, std::countr_zero(exp));
			ClearLSB(exp);
		}
		return sum;
	}
public:
	SumPow3Cache()
	{
		for (std::size_t i = 0; i < cache.size(); i++)
			cache[i] = sum_pow3(i);
	}
	int SumPow3(uint64_t exp) const noexcept { return cache[exp]; }
};

static SumPow3Cache sum_pow_3_cache;


int pown(int base, unsigned int exponent)
{
	int result = 1;
	while (exponent)
	{
		if (exponent % 2)
			result *= base;
		base *= base;
		exponent >>= 1;
	}
	return result;
}

uint32_t FastIndex(Position pos, uint64_t pattern) noexcept
{
	return sum_pow_3_cache.SumPow3(PExt(pos.Player(), pattern))
		+ sum_pow_3_cache.SumPow3(PExt(pos.Opponent(), pattern)) * 2;

	// 8 x AND, 4 x SHIFT, 4 x CMP, 5 x ADD
	// 1 x _mm256_reduce_add_epi32 (6 OPs)
	// = 27 OPs

	//int32x8 P_1(PExt(pos.Player(), pattern));
	//int32x8 P_2 = P_1;
	//int32x8 O_1(PExt(pos.Opponent(), pattern));
	//int32x8 O_2 = O_1;
	//const int32x8 iota_1(0, 1, 2, 3, 4, 5, 6, 7);
	//const int32x8 iota_2(8, 9, 10, 11, 12, 13, 14, 15);
	//const int32x8 pow3_1(1, 3, 9, 27, 81, 243, 729, 2187);
	//const int32x8 pow3_2(6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907);
	//const int32x8 one(1);

	//P_1 = (P_1 >> iota_1) & one;
	//P_2 = (P_2 >> iota_2) & one;
	//O_1 = (O_1 >> iota_1) & one;
	//O_2 = (O_2 >> iota_2) & one;

	//auto mask_1 = cmpeq(P_1, one);
	//auto mask_2 = cmpeq(P_2, one);
	//auto mask_3 = cmpeq(O_1, one);
	//auto mask_4 = cmpeq(O_2, one);

	//P_1 = mask_1 & pow3_1;
	//P_2 = mask_2 & pow3_2;
	//O_1 = mask_3 & pow3_1;
	//O_2 = mask_4 & pow3_2;
	//
	//return reduce_add(P_1 + P_2 + O_1 + O_1 + O_2 + O_2);
}

Configurations::Iterator::Iterator(uint64_t mask) noexcept
	: mask(mask)
	, size(1ULL << std::popcount(mask))
{}

Configurations::Iterator& Configurations::Iterator::operator++()
{
	o++;
	for (; p < size; p++) {
		for (; o < size; o++)
			if ((p & o) == 0u) // fields can only be taken by one player.
				return *this;
		o = 0;
	}
	*this = end(); // marks generator as depleted.
	return *this;
}

Position Configurations::Iterator::operator*() const
{
	return { PDep(p, mask), PDep(o, mask) };
}
