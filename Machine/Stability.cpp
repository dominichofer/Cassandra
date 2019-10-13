#include "Stability.h"
#include "Machine/BitTwiddling.h"
#include "Core/Machine.h"
#include "Core/Moves.h"
#include <array>

class StabilityAnalyzer
{
public:
	StabilityAnalyzer();

	uint64_t StableEdges(Position) const;
	uint64_t StableStones(Position) const; // Stable stones of the opponent.

private:
	static uint64_t FullLineHorizontal(uint64_t discs);
	static uint64_t FullLineVertival(uint64_t discs);
	static uint64_t FullLineDiagonal(uint64_t discs);
	static uint64_t FullLineCodiagonal(uint64_t discs);

	std::array<std::array<uint8_t, 256>, 256> edge_stables{};
};

StabilityAnalyzer::StabilityAnalyzer()
{
	for (std::size_t empty_count = 0; empty_count < 9; empty_count++)
		for (uint64_t p = 0; p < 256; p++)
			for (uint64_t o = 0; o < 256; o++)
			{
				if ((p & o) != 0u)
					continue;
				if (PopCount(p | o) != empty_count)
					continue;

				Board A{ BitBoard{ p }, BitBoard{ o } };
				Board B{ BitBoard{ o }, BitBoard{ p } };

				uint8_t stables = 0xFF;
				Moves moves(A.Empties() & BitBoard { 0xFFui64 });
				for (auto move : moves)
				{
					BitBoard A_flips = Flips(A, move);
					BitBoard B_flips = Flips(B, move);
					Board A_next = Play(A, move, A_flips);
					Board B_next = Play(B, move, B_flips);
					uint64_t A_acquired = A_next.O ^ A.P;
					uint64_t B_acquired = B_next.O ^ B.P;

					stables &= edge_stables[A_next.O][A_next.P] & ~A_acquired;
					stables &= edge_stables[B_next.O][B_next.P] & ~B_acquired;

					if (stables == 0)
						break;
				}
				edge_stables[p][o] = stables;
			}
}

uint64_t StabilityAnalyzer::StableEdges(Position pos) const
{
	// 2 x AND, 2 X SHIFT, 3 x OR, 4 x PEXT, 2 X PDEP
	// 13 OPs
	constexpr uint64_t L0_Left =
		"#              "
		"#              "
		"#              "
		"#              "
		"#              "
		"#              "
		"#              "
		"#              "_BitBoard;

	constexpr uint64_t L0_Right =
		"              #"
		"              #"
		"              #"
		"              #"
		"              #"
		"              #"
		"              #"
		"              #"_BitBoard;

	const auto stable_L0_Bottom = edge_stables[static_cast<uint8_t>(pos.GetP())][static_cast<uint8_t>(pos.GetO())];
	const auto stable_L0_Top = static_cast<uint64_t>(edge_stables[pos.GetP() >> 56][pos.GetO() >> 56]) << 56;
	const auto stable_L0_Left  = PDep(edge_stables[PExt(pos.GetP(), L0_Left )][PExt(pos.GetO(), L0_Left )], L0_Left );
	const auto stable_L0_Right = PDep(edge_stables[PExt(pos.GetP(), L0_Right)][PExt(pos.GetO(), L0_Right)], L0_Right);

	return stable_L0_Bottom | stable_L0_Top | stable_L0_Left | stable_L0_Right;
}

uint64_t StabilityAnalyzer::StableStones(Position pos) const
{
	const uint64_t discs = ~pos.Empties();

	const uint64_t full_h = FullLineHorizontal(discs);
	const uint64_t full_v = FullLineVertival(discs);
	const uint64_t full_d = FullLineDiagonal(discs);
	const uint64_t full_c = FullLineCodiagonal(discs);
	uint64_t new_stables = StableEdges(pos) & pos.GetO();
	new_stables |= full_h & full_v & full_d & full_c & pos.GetO() & ~BitBoard::Edge();

	uint64_t stables = 0;
	while ((new_stables & ~stables) != 0u)
	{
		stables |= new_stables;
		const uint64_t stables_h = (stables >> 1) | (stables << 1) | full_h;
		const uint64_t stables_v = (stables >> 8) | (stables << 8) | full_v;
		const uint64_t stables_d = (stables >> 9) | (stables << 9) | full_d;
		const uint64_t stables_c = (stables >> 7) | (stables << 7) | full_c;
		new_stables = stables_h & stables_v & stables_d & stables_c & pos.GetO() & ~BitBoard::Edge();
	}
	return stables;
}

uint64_t StabilityAnalyzer::FullLineHorizontal(uint64_t discs)
{
	// 4 x AND, 3 x SHIFT, 1 x MUL
	// 8 OPs
	discs &= discs >> 4;
	discs &= discs >> 2;
	discs &= discs >> 1;
	discs &= 0x0001010101010100ui64;
	return discs * 0xFFui64;
}

uint64_t StabilityAnalyzer::FullLineVertival(uint64_t discs)
{
	// 4 x AND, 3 x SHIFT, 1 x MUL
	// 8 OPs
	discs &= discs >> 32;
	discs &= discs >> 16;
	discs &= discs >> 8;
	discs &= 0x7Eui64;
	return discs * 0x0101010101010101ui64;
}

uint64_t StabilityAnalyzer::FullLineDiagonal(const uint64_t discs)
{
	// 5 x SHR, 5 x SHL, 7x AND, 10 x OR
	// 27 OPs
	constexpr uint64_t edge = BitBoard::Edge();

	uint64_t full_l = discs & (edge | (discs >> 9));
	uint64_t full_r = discs & (edge | (discs << 9));
	uint64_t edge_l = edge | (edge >> 9);
	uint64_t edge_r = edge | (edge << 9);
	full_l &= edge_l | (full_l >> 18);
	full_r &= edge_r | (full_r << 18);
	edge_l |= edge_l >> 18;
	edge_r |= edge_r << 18;
	full_l &= edge_l | (full_l >> 36);
	full_r &= edge_r | (full_r << 36);

	return full_r & full_l;
}

uint64_t StabilityAnalyzer::FullLineCodiagonal(const uint64_t discs)
{
	// 5 x SHR, 5 x SHL, 7x AND, 10 x OR
	// 27 OPs
	constexpr uint64_t edge = BitBoard::Edge();

	uint64_t full_l = discs & (edge | (discs >> 7));
	uint64_t full_r = discs & (edge | (discs << 7));
	uint64_t edge_l = edge | (edge >> 7);
	uint64_t edge_r = edge | (edge << 7);
	full_l &= edge_l | (full_l >> 14);
	full_r &= edge_r | (full_r << 14);
	edge_l |= edge_l >> 14;
	edge_r |= edge_r << 14;
	full_l &= edge_l | (full_l >> 28);
	full_r &= edge_r | (full_r << 28);

	return full_r & full_l;
}

static StabilityAnalyzer sa;

BitBoard StableEdges(Position pos)
{
	return BitBoard{ sa.StableEdges(pos) };
}

BitBoard StableStones(Position pos)
{
	return BitBoard{ sa.StableStones(pos) };
}