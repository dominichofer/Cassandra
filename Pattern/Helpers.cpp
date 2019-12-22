#include "Helpers.h"
#include <array>

// Forward declarations
[[nodiscard]] unsigned int BitScanLSB(uint64_t) noexcept;
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
	std::array<int, (1ULL << 15)> m_cache{};

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
	uint64_t SumPow3(uint64_t exp) const { return m_cache[exp]; }
};

static SumPow3Cache sum_pow_3_cache;

int FullIndex(const Position pos, const BitBoard pattern)
{
	return sum_pow_3_cache.SumPow3(PExt(pos.GetP(), pattern))
		+ sum_pow_3_cache.SumPow3(PExt(pos.GetO(), pattern)) * 2;
}

int ReducedIndex(const Position pos, const BitBoard pattern)
{
	return FullIndex(pos, pattern);
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
