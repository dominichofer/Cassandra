#pragma once
#include "BitBoard.h"
#include <cstdint>
#include <cstddef>

class Position;

struct Board
{
	BitBoard P, O;

	Board(Position);
	constexpr Board() noexcept : P(0), O(0) {}
	constexpr Board(BitBoard P, BitBoard O) noexcept : P(P), O(O) {}

	bool operator==(const Board& o) const { return (P == o.P) && (O == o.O); }
	bool operator!=(const Board& o) const { return (P != o.P) || (O != o.O); }

	BitBoard Empties() const { return ~(P | O); }
	std::size_t EmptyCount() const { return Empties().PopCount(); }
	
	uint64_t ParityQuadrants() const;

	void FlipCodiagonal();
	void FlipDiagonal();
	void FlipHorizontal();
	void FlipVertical();
};

inline Board FlipCodiagonal(Board b) { b.FlipCodiagonal(); return b; }
inline Board FlipDiagonal  (Board b) { b.FlipDiagonal  (); return b; }
inline Board FlipHorizontal(Board b) { b.FlipHorizontal(); return b; }
inline Board FlipVertical  (Board b) { b.FlipVertical  (); return b; }

class Position : private Board
{
public:
	constexpr Position() noexcept = default;
	constexpr Position(BitBoard P, BitBoard O) noexcept;
	Position(Board);

	static Position Start();
	static Position StartETH();

	using Board::Empties;
	using Board::EmptyCount;
	using Board::ParityQuadrants;

	//using Board::FlipCodiagonal;
	//using Board::FlipDiagonal;
	//using Board::FlipHorizontal;
	//using Board::FlipVertical;
	//using Board::FlipToMinimum;

	bool operator==(const Position& o) const { return Board::operator==(o); }
	bool operator!=(const Position& o) const { return Board::operator!=(o); }
	//bool operator<=(const Position& o) const { return Board::operator<=(o); }
	//bool operator>=(const Position& o) const { return Board::operator>=(o); }
	//bool operator< (const Position& o) const { return Board::operator< (o); }
	//bool operator> (const Position& o) const { return Board::operator> (o); }

	BitBoard GetP() const { return P; }
	BitBoard GetO() const { return O; }
};

//inline Position FlipCodiagonal(Position p) { p.FlipCodiagonal(); return p; }
//inline Position FlipDiagonal  (Position p) { p.FlipDiagonal  (); return p; }
//inline Position FlipHorizontal(Position p) { p.FlipHorizontal(); return p; }
//inline Position FlipVertical  (Position p) { p.FlipVertical  (); return p; }

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

Position operator""_pos(const char* c, std::size_t size);