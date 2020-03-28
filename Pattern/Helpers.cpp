#include "Helpers.h"
#include <array>


// TODO: Remove?
#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(__GNUC__)
	#include <x86intrin.h>
#else
	#error compiler not supported
#endif




// Forward declarations
[[nodiscard]] std::size_t BitScanLSB(uint64_t) noexcept;
void RemoveLSB(uint64_t&) noexcept;
[[nodiscard]] std::size_t PopCount(uint64_t) noexcept;
[[nodiscard]] uint64_t PDep(uint64_t src, uint64_t mask) noexcept;
[[nodiscard]] uint64_t PExt(uint64_t src, uint64_t mask) noexcept;


uint64_t Pow_int(uint64_t base, uint64_t exponent)
{
	if (exponent == 0)
		return 1;
	if (exponent % 2 == 0)
		return Pow_int(base * base, exponent / 2);

	return base * Pow_int(base, exponent - 1);
}

class SumPow3Cache
{
	std::array<int, (1ULL << 10)> m_cache{}; // 4kB

	int sum_pow3(uint64_t exp)
	{
		int sum = 0;
		while (exp != 0u)
		{
			sum += Pow_int(3, BitScanLSB(exp));
			RemoveLSB(exp);
		}
		return sum;
	}
public:
	SumPow3Cache()
	{
		for (std::size_t i = 0; i < std::size(m_cache); i++)
			m_cache[i] = sum_pow3(i);
	}
	uint64_t SumPow3(uint64_t exp) const noexcept { return m_cache[exp]; }
};

static SumPow3Cache sum_pow_3_cache;

uint32_t _mm256_reduce_add_epi32(__m256i x) noexcept;

int Index(const Position& pos, const BitBoard pattern) noexcept
{
	return sum_pow_3_cache.SumPow3(PExt(pos.P, pattern))
		+ sum_pow_3_cache.SumPow3(PExt(pos.O, pattern)) * 2;

	// 8 x AND, 4 x SHIFT, 4 x CMP, 5 x ADD
	// 1 x _mm256_reduce_add_epi32 (6 OPs)
	// = 27 OPs

	//auto P_1 = _mm256_set1_epi32(PExt(pos.P, pattern));
	//auto P_2 = P_1;
	//auto O_1 = _mm256_set1_epi32(PExt(pos.O, pattern));
	//auto O_2 = O_1;
	//const auto iota_1 = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	//const auto iota_2 = _mm256_set_epi32(8, 9, 10, 11, 12, 13, 14, 15);
	//const auto pow3_1 = _mm256_set_epi32(1, 3, 9, 27, 81, 243, 729, 2187);
	//const auto pow3_2 = _mm256_set_epi32(6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907);
	//const auto one = _mm256_set1_epi32(1);

	//P_1 = _mm256_and_si256(_mm256_srlv_epi32(P_1, iota_1), one);
	//P_2 = _mm256_and_si256(_mm256_srlv_epi32(P_2, iota_2), one);
	//O_1 = _mm256_and_si256(_mm256_srlv_epi32(O_1, iota_1), one);
	//O_2 = _mm256_and_si256(_mm256_srlv_epi32(O_2, iota_2), one);

	//auto mask_1 = _mm256_cmpeq_epi32(P_1, one);
	//auto mask_2 = _mm256_cmpeq_epi32(P_2, one);
	//auto mask_3 = _mm256_cmpeq_epi32(O_1, one);
	//auto mask_4 = _mm256_cmpeq_epi32(O_2, one);

	//P_1 = _mm256_and_si256(mask_1, pow3_1);
	//P_2 = _mm256_and_si256(mask_2, pow3_2);
	//O_1 = _mm256_and_si256(mask_3, pow3_1);
	//O_2 = _mm256_and_si256(mask_4, pow3_2);

	//O_1 = _mm256_add_epi32(O_1, O_1); // simulates a "*2".
	//O_2 = _mm256_add_epi32(O_2, O_2); // simulates a "*2".
	//
	//auto x = _mm256_add_epi32(_mm256_add_epi32(P_1, P_2), _mm256_add_epi32(O_1, O_2));
	//return _mm256_reduce_add_epi32(x);
}

void For_each_config(const BitBoard pattern, const std::function<void(Position)>& fkt)
{
	const std::size_t size = 1ULL << PopCount(pattern);
	for (uint64_t i = 0; i < size; i++)
	{
		const BitBoard P{ PDep(i, pattern) };
		for (uint64_t j = 0; j < size; j++)
		{
			if ((i & j) != 0u)
				continue; // fields can only be taken by one player

			const BitBoard O{ PDep(j, pattern) };

			fkt(Position::TryCreate(P, O));
		}
	}
}
