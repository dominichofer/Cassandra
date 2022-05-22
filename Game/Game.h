#pragma once
#include "Puzzle.h"
#include <span>
#include <vector>

class Game
{
	Position pos;
	std::vector<Field> moves; // front is first move
public:
	class Sentinel {};
	class Iterator
	{
		Position pos;
		std::vector<Field>::const_iterator it, end;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Position;
		using reference = Position&;
		using pointer = Position*;
		using iterator_category = std::forward_iterator_tag;

		Iterator(Position pos, const std::vector<Field>& moves) noexcept : pos(pos), it(moves.begin()), end(moves.end()) {}

		bool operator==(const Iterator&) const noexcept = default;
		bool operator!=(const Iterator&) const noexcept = default;
		bool operator==(const Sentinel&) const noexcept { return pos == Position{}; }
		bool operator!=(const Sentinel&) const noexcept { return pos != Position{}; }

		Iterator& operator++() noexcept
		{
			if (it == end)
			{
				pos = Position{};
				return *this;
			}
			pos = Play(pos, *it++);
			if (HasMoves(pos))
				return *this;
			Position passed = PlayPass(pos);
			if (HasMoves(passed))
				pos = passed;
			return *this;
		}
		Position operator*() const noexcept { return pos; }
	};

	Game(Position pos = Position::Start()) noexcept : pos(pos) {}
	Game(std::vector<Field> moves, Position pos = Position::Start()) noexcept : pos(pos), moves(std::move(moves)) {}

	void Add(Field move) { moves.push_back(move); }
	void clear() noexcept { moves.clear(); }
	bool empty() const noexcept { return moves.empty(); }
	std::size_t size() noexcept { return moves.size(); }

	Iterator begin() const noexcept { return { pos, moves }; }
	Sentinel end() const noexcept { return {}; }
};

// Game Generators
namespace GameGen
{
	// Abstract
	class GameGenerator
	{
	public:
		virtual Game operator()() noexcept = 0;
	};

	class Random final : public GameGenerator
	{
		std::mt19937_64 rnd_engine;
	public:
		Random(unsigned int seed = std::random_device{}()) : rnd_engine(seed) {}

		Game operator()() noexcept override;
	};

	// Generator of played Position with given empty count.
	class Played : public GameGenerator
	{
	protected:
		Player& first, &second;
		std::vector<Position> start;
		std::mt19937_64 rnd_engine{ std::random_device{}() };
		std::uniform_int_distribution<std::size_t> start_pick;
	public:
		Played(Player& first, Player& second, std::vector<Position> start);
		Played(Player& first, Player& second, Position start = Position::Start()) : Played(first, second, std::vector{ start }) {}

		Game operator()() noexcept override;
	};
}