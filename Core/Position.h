#pragma once
#include "Bit.h"
#include "BitBoard.h"
#include "Moves.h"
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

inline int min_score = -32;
inline int max_score = +32;
inline int inf_score = +33;
inline int undefined_score = +35;

template <typename T>
concept score_range = std::ranges::range<T> and std::is_same_v<std::ranges::range_value_t<T>, int>; // TODO: Is this used?

// Maps input to (.., "-1", "+0", "+1", ..)
std::string SignedInt(int);

// Maps input to (.., "-01", "+00", "+01", ..)
std::string DoubleDigitSignedInt(int);


// A board where every field is either taken by exactly one player or is empty.
class Position
{
	BitBoard P{}, O{};
public:
	constexpr Position() noexcept = default;
	CUDA_CALLABLE constexpr Position(BitBoard P, BitBoard O) noexcept : P(P), O(O) { assert((P & O).empty()); }

	static Position Start();
	static Position StartETH();

	static constexpr Position From(BitBoard P, BitBoard O) noexcept(false)
	{
		if ((P & O).empty())
			return { P, O };
		throw;
	}

	//[[nodiscard]] constexpr auto operator<=>(const Position&) const noexcept = default;
	[[nodiscard]] CUDA_CALLABLE constexpr bool operator==(const Position& o) const noexcept { return std::tie(P, O) == std::tie(o.P, o.O); }
	[[nodiscard]] CUDA_CALLABLE constexpr bool operator!=(const Position& o) const noexcept { return std::tie(P, O) != std::tie(o.P, o.O); }
	[[nodiscard]] CUDA_CALLABLE constexpr bool operator<(const Position& o) const noexcept { return std::tie(P, O) < std::tie(o.P, o.O); }

	[[nodiscard]] CUDA_CALLABLE BitBoard Player() const noexcept { return P; }
	[[nodiscard]] CUDA_CALLABLE BitBoard Opponent() const noexcept { return O; }

	CUDA_CALLABLE void FlipCodiagonal() noexcept;
	CUDA_CALLABLE void FlipDiagonal() noexcept;
	CUDA_CALLABLE void FlipHorizontal() noexcept;
	CUDA_CALLABLE void FlipVertical() noexcept;
	CUDA_CALLABLE void FlipToUnique() noexcept;

	[[nodiscard]] CUDA_CALLABLE BitBoard Discs() const { return P | O; }
	[[nodiscard]] CUDA_CALLABLE BitBoard Empties() const { return ~Discs(); }
	[[nodiscard]] CUDA_CALLABLE int EmptyCount() const { return popcount(Empties()); }

	[[nodiscard]] BitBoard ParityQuadrants() const;
};

template <typename T>
concept pos_range = std::ranges::range<T> and std::is_same_v<std::ranges::range_value_t<T>, Position>;

[[nodiscard]] CUDA_CALLABLE Position FlipCodiagonal(Position) noexcept;
[[nodiscard]] CUDA_CALLABLE Position FlipDiagonal(Position) noexcept;
[[nodiscard]] CUDA_CALLABLE Position FlipHorizontal(Position) noexcept;
[[nodiscard]] CUDA_CALLABLE Position FlipVertical(Position) noexcept;
[[nodiscard]] CUDA_CALLABLE Position FlipToUnique(Position) noexcept;


// TODO: Add tests for these 6 and the Neighbours!
//[[nodiscard]] CUDA_CALLABLE inline BitBoard PotentialMoves(const Position& pos) noexcept { return pos.Empties() & EightNeighboursAndSelf(pos.Opponent()); }
//[[nodiscard]] CUDA_CALLABLE inline BitBoard PotentialCounterMoves(const Position& pos) noexcept { return pos.Empties() & EightNeighboursAndSelf(pos.Player()); }
//
//[[nodiscard]] CUDA_CALLABLE inline BitBoard ExposedPlayers(const Position& pos) noexcept { return pos.Player() & EightNeighboursAndSelf(pos.Empties()); }
//[[nodiscard]] CUDA_CALLABLE inline BitBoard ExposedOpponents(const Position& pos) noexcept { return pos.Opponent() & EightNeighboursAndSelf(pos.Empties()); }
//
//[[nodiscard]] CUDA_CALLABLE inline BitBoard IsolatedPlayers(const Position& pos) noexcept { return pos.Player() & ~EightNeighboursAndSelf(pos.Empties()); }
//[[nodiscard]] CUDA_CALLABLE inline BitBoard IsolatedOpponents(const Position& pos) noexcept { return pos.Opponent() & ~EightNeighboursAndSelf(pos.Empties()); }

[[nodiscard]] std::string SingleLine(const Position&);
[[nodiscard]] std::string MultiLine(const Position&);
[[nodiscard]] inline std::string to_string(const Position& pos) { return SingleLine(pos); }
inline std::ostream& operator<<(std::ostream& os, const Position& pos) { return os << to_string(pos); }

[[nodiscard]]
constexpr Position operator""_pos(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard P{0};
	BitBoard O{0};
	for (int j = 0; j < 8; j++)
		for (int i = 0; i < 8; i++)
		{
			char symbol = c[119 - 15 * j - 2 * i];
			if (symbol == 'X')
				P.Set(i, j);
			if (symbol == 'O')
				O.Set(i, j);
		}
	return { P, O };
}

[[nodiscard]]
CUDA_CALLABLE int EvalGameOver(const Position&) noexcept;

[[nodiscard]]
CUDA_CALLABLE Position Play(const Position&, Field move, BitBoard flips);

[[nodiscard]]
CUDA_CALLABLE Position Play(const Position&, Field move);

[[nodiscard]]
CUDA_CALLABLE Position PlayPass(const Position&) noexcept;

[[nodiscard]]
CUDA_CALLABLE BitBoard Flips(const Position&, Field move) noexcept;

[[nodiscard]]
int CountLastFlip(const Position&, Field move) noexcept;

[[nodiscard]]
BitBoard StableEdges(const Position&);

// Stable stones of the opponent.
[[nodiscard]]
BitBoard StableStonesOpponent(const Position&);

// Stable corners + 1 of the opponent.
[[nodiscard]]
BitBoard StableCornersOpponent(const Position&);

[[nodiscard]]
int StabilityBasedMaxScore(const Position&);

[[nodiscard]]
CUDA_CALLABLE Moves PossibleMoves(const Position&) noexcept;

namespace detail
{
	#ifdef __AVX512F__
		[[nodiscard]]
		Moves PossibleMoves_AVX512(const Position&) noexcept;
	#endif

	#ifdef __AVX2__
		[[nodiscard]]
		Moves PossibleMoves_AVX2(const Position&) noexcept;
	#endif

	[[nodiscard]]
	CUDA_CALLABLE Moves PossibleMoves_x64(const Position&) noexcept;
}

[[nodiscard]]
CUDA_CALLABLE bool HasMoves(const Position&) noexcept;

namespace detail
{
	#ifdef __AVX512F__
		[[nodiscard]]
		bool HasMoves_AVX512(const Position&) noexcept;
	#endif

	#ifdef __AVX2__
		[[nodiscard]]
		bool HasMoves_AVX2(const Position&) noexcept;
	#endif

	[[nodiscard]]
	CUDA_CALLABLE bool HasMoves_x64(const Position&) noexcept;
}