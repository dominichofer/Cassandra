#include "Position.h"
#include "Machine/BitTwiddling.h"

void Board::FlipCodiagonal() { P.FlipCodiagonal(); O.FlipCodiagonal(); }
void Board::FlipDiagonal  () { P.FlipDiagonal  (); O.FlipDiagonal  (); }
void Board::FlipHorizontal() { P.FlipHorizontal(); O.FlipHorizontal(); }
void Board::FlipVertical  () { P.FlipVertical  (); O.FlipVertical  (); }

//void Board::FlipToMinimum()
//{
//	Board copy = *this;
//	copy.FlipVertical();		if (copy < *this) *this = copy;
//	copy.FlipHorizontal();		if (copy < *this) *this = copy;
//	copy.FlipVertical();		if (copy < *this) *this = copy;
//	copy.FlipDiagonal();		if (copy < *this) *this = copy;
//	copy.FlipVertical();		if (copy < *this) *this = copy;
//	copy.FlipHorizontal();		if (copy < *this) *this = copy;
//	copy.FlipVertical();		if (copy < *this) *this = copy;
//}

Board::Board(Position pos) : P(pos.GetP()), O(pos.GetO()) {}

uint64_t Board::ParityQuadrants() const
{
	// 4 x SHIFT, 4 x XOR, 1 x AND, 1 x NOT, 1x OR, 1 x MUL
	// = 12 OPs
	uint64_t E = Empties();
	E ^= E >> 1;
	E ^= E >> 2;
	E ^= E >> 8;
	E ^= E >> 16;
	E &= 0x0000'0011'0000'0011ui64;
	return E * 0x0000'0000'0F0F'0F0Fui64;
}


constexpr Position::Position(BitBoard P, BitBoard O) noexcept : Board{ P, O }
{
	assert((P & O) == 0); // Only one stone per field.
	assert(BitBoard::Middle().isSubsetOf(~Empties())); // middle fields have to be taken.
}

Position::Position(Board b) : Position(b.P, b.O)
{}

Position Position::Start()
{
	return
		"               "
		"               "
		"               "
		"      O X      "
		"      X O      "
		"               "
		"               "
		"               "_pos;
}

Position Position::StartETH()
{
	return
		"               "
		"               "
		"               "
		"      X X      "
		"      O O      "
		"               "
		"               "
		"               "_pos;
}

Position FlipToUnique(Position pos)
{
	auto less = [](Board l, Board r) {
		if (l.P == r.P)
			return static_cast<uint64_t>(l.O) < static_cast<uint64_t>(r.O);
		else
			return static_cast<uint64_t>(l.P) < static_cast<uint64_t>(r.P);
	};

	Board candidate = pos;
	Board min = candidate;
	candidate.FlipVertical();		if (less(candidate, min)) min = candidate;
	candidate.FlipHorizontal();		if (less(candidate, min)) min = candidate;
	candidate.FlipVertical();		if (less(candidate, min)) min = candidate;
	candidate.FlipCodiagonal();		if (less(candidate, min)) min = candidate;
	candidate.FlipVertical();		if (less(candidate, min)) min = candidate;
	candidate.FlipHorizontal();		if (less(candidate, min)) min = candidate;
	candidate.FlipVertical();		if (less(candidate, min)) min = candidate;
	return min;
}

Position operator""_pos(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard P(0);
	BitBoard O(0);
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
		{
			auto field = static_cast<Field>(i * 8 + j);
			char symbol = c[119 - 2 * j - 15 * i];
			if (symbol == 'X')
				P[field] = true;
			else if (symbol == 'O')
				O[field] = true;
		}
	return { P, O };
}