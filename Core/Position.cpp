#include "Position.h"
#include "Machine/BitTwiddling.h"

 Board::Board(const Position& pos) : P(pos.GetP()), O(pos.GetO()) {}

void Board::FlipCodiagonal() { P = ::FlipCodiagonal(P); O = ::FlipCodiagonal(O); }
void Board::FlipDiagonal  () { P = ::FlipDiagonal  (P); O = ::FlipDiagonal  (O); }
void Board::FlipHorizontal() { P = ::FlipHorizontal(P); O = ::FlipHorizontal(O); }
void Board::FlipVertical  () { P = ::FlipVertical  (P); O = ::FlipVertical  (O); }

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


constexpr Position::Position(uint64_t P, uint64_t O) : Board{ P, O }
{
	assert((P & O) == 0);
	assert(TestBits(~Empties(), Board::middle));
}

Position::Position(Board b) : Position(b.P, b.O)
{}

Position Position::Start() { return Position(0x0000'0008'1000'0000ui64, 0x0000'0010'0800'0000ui64); }
Position Position::StartETH() { return Position(0x0000'0018'0000'0000ui64, 0x0000'0000'1800'0000ui64); }