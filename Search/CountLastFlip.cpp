#include "CountLastFlip.h"
#include <array>
#include <cstdint>

uint64_t HorizontalLine(uint8_t field) noexcept
{
	return 0xFFULL << (field & 0xF8);
}

uint64_t VerticalLine(uint8_t field) noexcept
{
	return 0x0101010101010101ULL << (field % 8);
}

uint64_t DiagonalLine(uint8_t field) noexcept
{
	int offset = (field / 8) - (field % 8);
	if (offset > 0)
		return 0x8040201008040201ULL << (offset * 8); // upper half
	else
		return 0x8040201008040201ULL >> (-offset * 8); // lower half
}

uint64_t CodiagonalLine(uint8_t field) noexcept
{
	int offset = (field / 8) + (field % 8) - 7;
	if (offset > 0)
		return 0x0102040810204080ULL << (offset * 8); // upper half
	else
		return 0x0102040810204080ULL >> (-offset * 8); // lower half
}


class CLF
{
	std::array<std::array<uint64_t, 4>, 64> mask;
	std::array<std::array<uint8_t, 256>, 8> flip_count;
public:
	CLF() noexcept
	{
		for (uint8_t move = 0; move < 64; move++)
		{
			const uint64_t relevant = HorizontalLine(move) | VerticalLine(move) | DiagonalLine(move) | CodiagonalLine(move);
			
			uint64_t irrelevant = ~relevant;
			uint64_t codiagonal = CodiagonalLine(move);
			while (PExt(1ULL << move, codiagonal) != 1ULL << (move / 8))
			{
				codiagonal |= GetLSB(irrelevant);
				RemoveLSB(irrelevant);
			}

			irrelevant = ~relevant;
			uint64_t diagonal = DiagonalLine(move);
			while (PExt(1ULL << move, diagonal) != 1ULL << (move / 8))
			{
				diagonal |= GetLSB(irrelevant);
				RemoveLSB(irrelevant);
			}

			mask[move][0] = relevant;
			mask[move][1] = codiagonal;
			mask[move][2] = diagonal;
			mask[move][3] = VerticalLine(move);
		}

		for (uint8_t move = 0; move < 8; move++)
			for (uint64_t r = 0; r < 256; r++)
			{
				uint64_t mask = ~(1ULL << move);
				Position pos(r & mask, ~r & mask);
				flip_count[move][r] = std::popcount(Flips(pos, static_cast<Field>(move)));
			}
	}

	int Count(const Position& pos, Field f) const noexcept
	{
		auto move = std::to_underlying(f);
		auto x = move % 8;
		auto y = move / 8;

		auto P = pos.Player() & mask[move][0]; // mask out unrelated bits to make them dummy 0 bits

		return flip_count[x][BExtr(P, move & 0xF8, 8)]
			+ flip_count[y][PExt(P, mask[move][1])]
			+ flip_count[y][PExt(P, mask[move][2])]
			+ flip_count[y][PExt(P, mask[move][3])];
	}
};

static CLF clf;

int CountLastFlip(const Position& pos, Field f) noexcept
{
	return clf.Count(pos, f);
}
