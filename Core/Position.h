#pragma once
#include "BitBoard.h"
#include <cstdint>
#include <cstddef>
#include <compare>


// A board where every field is either taken by a player or empty.
class Position
{
	BitBoard P, O;
public:
	constexpr Position() noexcept = default;
	constexpr Position(BitBoard P, BitBoard O) noexcept : P(P), O(O) { assert(MeetsConstraints(P, O)); }

	static Position Start();
	static Position StartETH();

	static constexpr Position TryCreate(BitBoard P, BitBoard O) noexcept(false)
	{
		if (MeetsConstraints(P, O))
			return { P, O };
		throw;
	}

	static constexpr bool MeetsConstraints(BitBoard P, BitBoard O) noexcept
	{
		return (P & O).empty();
	}

	const BitBoard& GetP() const & { return P; }
	const BitBoard& GetO() const & { return O; }
	BitBoard&& GetP() && { return std::move(P); }
	BitBoard&& GetO() && { return std::move(O); }

	[[nodiscard]] auto operator<=>(const Position&) const noexcept = default;

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
	void FlipToUnique() noexcept;

	BitBoard Empties() const { return ~(P | O); }
	std::size_t EmptyCount() const { return Empties().PopCount(); }
	
	uint64_t ParityQuadrants() const;
};

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

Position operator""_pos(const char*, std::size_t size);


Position FlipCodiagonal(Position pos) noexcept;
Position FlipDiagonal(Position pos) noexcept;
Position FlipHorizontal(Position pos) noexcept;
Position FlipVertical(Position pos) noexcept;
Position FlipToUnique(Position pos) noexcept;