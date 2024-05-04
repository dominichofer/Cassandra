#include "Stability.h"
#include "Game/Game.h"
#include <array>

class StabilityAnalyzer
{
public:
	StabilityAnalyzer();

	uint64_t StableEdges(const Position&) const; // Stable edges of player and opponent
	uint64_t StableStonesOpponent(const Position&) const; // Stable stones of the opponent

private:
	static uint64_t FullLineHorizontal(uint64_t discs);
	static uint64_t FullLineVertival(uint64_t discs);
	static uint64_t FullLineDiagonal(uint64_t discs);
	static uint64_t FullLineCodiagonal(uint64_t discs);

	// edge_stables[Player][Opponent] holds the stable discs of player and opponent
	std::array<std::array<uint8_t, 256>, 256> edge_stables{};
};

StabilityAnalyzer::StabilityAnalyzer()
{
	for (int empty_count = 56; empty_count <= 64; empty_count++)
		for (uint64_t p = 0; p < 256; p++)
			for (uint64_t o = 0; o < 256; o++)
			{
				if (p & o)
					continue;

				Position pos{ p, o };
				if (pos.EmptyCount() != empty_count)
					continue;

				const Position A{ p, o };
				const Position B{ o, p };

				uint8_t stables = 0xFF;
				for (const auto& move : Moves(~(p | o) & 0xFFULL))
				{
					Position A_next = Play(A, move);
					Position B_next = Play(B, move);
					uint64_t A_unchanged = ~(A.Player() ^ A_next.Opponent());
					uint64_t B_unchanged = ~(B.Player() ^ B_next.Opponent());

					stables &= edge_stables[A_next.Opponent()][A_next.Player()] & A_unchanged;
					stables &= edge_stables[B_next.Opponent()][B_next.Player()] & B_unchanged;

					if (stables == 0)
						break;
				}
				edge_stables[p][o] = stables;
			}
}

uint64_t StabilityAnalyzer::StableEdges(const Position& pos) const
{	
	constexpr uint64_t L0_Left = 0x8080808080808080ULL;
	constexpr uint64_t L0_Right = 0x0101010101010101ULL;

	const uint64_t P = pos.Player();
	const uint64_t O = pos.Opponent();
	const auto stable_L0_Bottom = edge_stables[P & 0xFF][O & 0xFF];
	const auto stable_L0_Top = static_cast<uint64_t>(edge_stables[P >> 56][O >> 56]) << 56;
	const auto stable_L0_Left  = PDep(edge_stables[PExt(P, L0_Left )][PExt(O, L0_Left )], L0_Left );
	const auto stable_L0_Right = PDep(edge_stables[PExt(P, L0_Right)][PExt(O, L0_Right)], L0_Right);

	return stable_L0_Bottom | stable_L0_Top | stable_L0_Left | stable_L0_Right;
}

uint64_t StabilityAnalyzer::StableStonesOpponent(const Position& pos) const
{
	const uint64_t discs = pos.Discs();

	const uint64_t full_h = FullLineHorizontal(discs);
	const uint64_t full_v = FullLineVertival(discs);
	const uint64_t full_d = FullLineDiagonal(discs);
	const uint64_t full_c = FullLineCodiagonal(discs);
	uint64_t new_stables = StableEdges(pos) & pos.Opponent();
	new_stables |= full_h & full_v & full_d & full_c & pos.Opponent() & 0x007E7E7E7E7E7E00ULL;

	uint64_t stables = 0;
	while (new_stables & ~stables) // discovered new stables
	{
		stables |= new_stables;
		uint64_t stables_h = (stables >> 1) | (stables << 1) | full_h;
		uint64_t stables_v = (stables >> 8) | (stables << 8) | full_v;
		uint64_t stables_d = (stables >> 9) | (stables << 9) | full_d;
		uint64_t stables_c = (stables >> 7) | (stables << 7) | full_c;
		new_stables = stables_h & stables_v & stables_d & stables_c & pos.Opponent() & 0x007E7E7E7E7E7E00ULL;
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
	discs &= 0x0001010101010100ULL;
	return discs * 0xFFULL;
}

uint64_t StabilityAnalyzer::FullLineVertival(uint64_t discs)
{
	// 4 x AND, 3 x SHIFT, 1 x MUL
	// 8 OPs
	discs &= discs >> 32;
	discs &= discs >> 16;
	discs &= discs >> 8;
	discs &= 0x7EULL;
	return discs * 0x0101010101010101ULL;
}

uint64_t StabilityAnalyzer::FullLineDiagonal(const uint64_t discs)
{
	// 5 x SHR, 5 x SHL, 7x AND, 10 x OR
	// 27 OPs
	constexpr uint64_t edge = 0xFF818181818181FFULL;

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
	constexpr uint64_t edge = 0xFF818181818181FFULL;

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

uint64_t StableEdges(const Position& pos)
{
	return sa.StableEdges(pos);
}

uint64_t StableStonesOpponent(const Position& pos)
{
	return sa.StableStonesOpponent(pos);
}
