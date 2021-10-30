#pragma once
#include "Position.h"
#include "Moves.h"
#include "Player.h"

#include <random>
#include <iterator>
#include <vector>
#include <set>

// Position Generators
namespace PosGen
{
	// Abstract
	class PositionGenerator
	{
	public:
		virtual Position operator()() noexcept = 0;
	};

	class Random final : public PositionGenerator
	{
		std::mt19937_64 rnd_engine;
		std::uniform_int_distribution<uint64_t> dist{ 0, -1 };
	public:
		Random(unsigned int seed = std::random_device{}()) : rnd_engine(seed) {}

		Position operator()() noexcept override;
	};

	// Generator of random Position with given empty count.
	class RandomWithEmptyCount final : public PositionGenerator
	{
		const int empty_count;
		std::mt19937_64 rnd_engine;
		std::uniform_int_distribution<int> boolean{0, 1};
	public:
		RandomWithEmptyCount(int empty_count, unsigned int seed = std::random_device{}()) : empty_count(empty_count), rnd_engine(seed) {}
		
		Position operator()() noexcept override;
	};
	
	// Generator of played Position with given empty count.
	class Played : public PositionGenerator
	{
	protected:
		Player &first, &second;
		const int empty_count;
		std::vector<Position> start;
		std::mt19937_64 rnd_engine{ std::random_device{}() };
		std::uniform_int_distribution<int> start_pick;
	public:
		Played(Player& first, Player& second, int empty_count, std::vector<Position> start) noexcept(false);
		Played(Player& first, Player& second, int empty_count, Position start = Position::Start()) noexcept(false) : Played(first, second, empty_count, std::vector{ start }) {}

		Position operator()() noexcept override;
	};

	// Generator of randomly played Position with given empty count.
	class RandomlyPlayed final : public Played
	{
		RandomPlayer first, second;
	public:
		RandomlyPlayed(unsigned int seed1, unsigned int seed2, int empty_count, Position start = Position::Start()) noexcept
			: first(seed1), second(seed2), Played(first, second, empty_count, start)
		{}
		RandomlyPlayed(int empty_count, Position start = Position::Start()) noexcept
			: Played(first, second, empty_count, start) {}
	};

	std::set<Position> generate_n_unique(int count, PositionGenerator&);
	std::set<Position> generate_n_unique(int count, PositionGenerator&&);
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
			[[nodiscard]] bool operator==(const PosMov&) const noexcept = default;
			[[nodiscard]] bool operator!=(const PosMov&) const noexcept = default;
			//[[nodiscard]] bool operator==(const PosMov& o) const noexcept { return std::tie(pos, moves) == std::tie(o.pos, o.moves); }
			//[[nodiscard]] bool operator!=(const PosMov& o) const noexcept { return !(*this == o); }
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

std::vector<Position> AllUnique(Position, int empty_count);