#include "Core/Core.h"
#include "Helpers.h"
#include <array>

class SumPow3Cache
{
	std::array<int, (1ULL << 16)> m_cache{};

	static int sum_pow3(uint64 exp)
	{
		int sum = 0;
		while (exp != 0U)
		{
			sum += pown(3, countr_zero(exp));
			RemoveLSB(exp);
		}
		return sum;
	}
public:
	SumPow3Cache()
	{
		for (std::size_t i = 0; i < m_cache.size(); i++)
			m_cache[i] = sum_pow3(i);
	}
	int SumPow3(uint64 exp) const noexcept { return m_cache[exp]; }
};

static SumPow3Cache sum_pow_3_cache;

int FastIndex(const Position& pos, const BitBoard pattern) noexcept
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
