#pragma once
#include "Position.h"
#include "Moves.h"
#include "Player.h"

#include <random>
#include <iterator>
#include <vector>

// Position Generator
namespace PosGen
{
	// Generator of random Position.
	class Random
	{
		std::mt19937_64 rnd_engine;
	public:
		Random(unsigned int seed = std::random_device{}()) : rnd_engine(seed) {}

		Position operator()();
		std::vector<Position> operator()(int num);
	};

	// Generator of random Position with given empty count.
	class RandomWithEmptyCount
	{
		const int empty_count;
		std::mt19937_64 rnd_engine;
	public:
		RandomWithEmptyCount(int empty_count, unsigned int seed = std::random_device{}()) : empty_count(empty_count), rnd_engine(seed) {}
		
		Position operator()();
		std::vector<Position> operator()(int num);
	};
	
	// Generator of played Position with given empty count.
	class Played
	{
	protected:
		Player &first, &second;
		const int empty_count;
		const Position start;
	public:
		Played(Player& first, Player& second, int empty_count, Position start = Position::Start());

		Position operator()();
		std::vector<Position> operator()(int num);
	};

	// Generator of randomly played Position with given empty count.
	class RandomPlayed final : public Played
	{
		RandomPlayer rp_first, rp_second;
	public:
		RandomPlayed(unsigned int seed1, unsigned int seed2, int empty_count, Position start = Position::Start()) noexcept
			: rp_first(seed1), rp_second(seed2), Played(rp_first, rp_second, empty_count, start)
		{}
		RandomPlayed(int empty_count, Position start = Position::Start()) noexcept
			: Played(rp_first, rp_second, empty_count, start) {}
	};
}

class ChildrenGenerator
{
	class Iterator
	{
		struct PosMov
		{
			Position pos;
			Moves moves;

			PosMov(Position pos, Moves moves) : pos(pos), moves(moves) {}
			[[nodiscard]] bool operator==(const PosMov& o) const noexcept { return std::tie(pos, moves) == std::tie(o.pos, o.moves); }
			[[nodiscard]] bool operator!=(const PosMov& o) const noexcept { return !(*this == o); }
		};
		const int plies = 0;
		const bool pass_is_a_ply = false;
		std::vector<PosMov> stack{};
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		Iterator() noexcept = default;
		Iterator(const Position& start, int plies, bool pass_is_a_ply) noexcept;

		[[nodiscard]] bool operator==(const Iterator& o) const noexcept { return (stack.empty() && o.stack.empty()) || std::tie(plies, pass_is_a_ply, stack) == std::tie(o.plies, o.pass_is_a_ply, o.stack); }
		[[nodiscard]] bool operator!=(const Iterator& o) const noexcept { return !(*this == o); }
		Iterator& operator++();
		[[nodiscard]] const Position& operator*() const noexcept { return stack.back().pos; }
	};

	const Position start;
	const int plies;
	const bool pass_is_a_ply;
public:
	ChildrenGenerator(const Position& start, int plies, bool pass_is_a_ply) noexcept
		: start(start), plies(plies), pass_is_a_ply(pass_is_a_ply) {}

	[[nodiscard]] Iterator begin() const { return {start, plies, pass_is_a_ply}; }
	[[nodiscard]] Iterator cbegin() const { return {start, plies, pass_is_a_ply}; }
	[[nodiscard]] static Iterator end() { return {}; }
	[[nodiscard]] static Iterator cend() { return {}; }
};

ChildrenGenerator Children(Position, int plies, bool pass_is_a_ply);
ChildrenGenerator Children(Position, int empty_count);