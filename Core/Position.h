#pragma once
#include "Machine/BitTwiddling.h"

class Position;

struct Board
{
	uint64_t P, O;

	explicit Board(const Position&);
	constexpr Board(uint64_t P, uint64_t O) : P(P), O(O) {}

	uint64_t Empties() const { return ~(P | O); }
	std::size_t EmptyCount() const { return PopCount(Empties()); }
	
	uint64_t ParityQuadrants() const;

	void FlipCodiagonal();
	void FlipDiagonal();
	void FlipHorizontal();
	void FlipVertical();
	//void FlipToMinimum();

	bool operator==(const Board& o) const { return (P == o.P) && (O == o.O); }
	bool operator!=(const Board& o) const { return (P != o.P) || (O != o.O); }
	//bool operator<=(const Board& o) const { return (P == o.P) ? (O <= o.O) : (P <= o.P); }
	//bool operator>=(const Board& o) const { return (P == o.P) ? (O >= o.O) : (P >= o.P); }
	//bool operator< (const Board& o) const { return (P == o.P) ? (O < o.O) : (P < o.P); }
	//bool operator> (const Board& o) const { return (P == o.P) ? (O > o.O) : (P > o.P); }

	static constexpr uint64_t middle = 0x0000'0018'1800'0000ui64;
};

inline Board FlipCodiagonal(Board b) { b.FlipCodiagonal(); return b; }
inline Board FlipDiagonal  (Board b) { b.FlipDiagonal  (); return b; }
inline Board FlipHorizontal(Board b) { b.FlipHorizontal(); return b; }
inline Board FlipVertical  (Board b) { b.FlipVertical  (); return b; }
//inline Board FlipToMinimum (Board b) { b.FlipToMinimum (); return b; }

class Position : private Board
{
public:
	constexpr Position(uint64_t P, uint64_t O);
	Position(Board);

	static Position Start();
	static Position StartETH();

	using Board::Empties;
	using Board::EmptyCount;
	using Board::ParityQuadrants;

	using Board::FlipCodiagonal;
	using Board::FlipDiagonal;
	using Board::FlipHorizontal;
	using Board::FlipVertical;
	//using Board::FlipToMinimum;

	bool operator==(const Position& o) const { return Board::operator==(o); }
	bool operator!=(const Position& o) const { return Board::operator!=(o); }
	//bool operator<=(const Position& o) const { return Board::operator<=(o); }
	//bool operator>=(const Position& o) const { return Board::operator>=(o); }
	//bool operator< (const Position& o) const { return Board::operator< (o); }
	//bool operator> (const Position& o) const { return Board::operator> (o); }

	uint64_t GetP() const { return P; }
	uint64_t GetO() const { return O; }
};

inline Position FlipCodiagonal(Position p) { p.FlipCodiagonal(); return p; }
inline Position FlipDiagonal  (Position p) { p.FlipDiagonal  (); return p; }
inline Position FlipHorizontal(Position p) { p.FlipHorizontal(); return p; }
inline Position FlipVertical  (Position p) { p.FlipVertical  (); return p; }

Position FlipToUnique(Position pos);

//#include <functional>
//namespace std
//{
//	template<> struct hash<Position>
//	{
//		std::size_t operator()(const Position& pos) const
//		{
//			uint64_t P = pos.GetP();
//			uint64_t O = pos.GetO();
//			P ^= P >> 36;
//			O ^= O >> 21;
//			return P * O;
//		}
//	};
//}


constexpr Position operator""_pos(const char* c, std::size_t size)
{
	if (size != 64)
		throw "Invalid length of Position string literal";
	uint64_t P = 0;
	uint64_t O = 0;
	for (int i = 0; i < 64; i++)
	{
		if (c[63 - i] == 'X')
			SetBit(P, i);
		else if (c[63 - i] == 'O')
			SetBit(O, i);
	}
	return { P,O };
}
