#pragma once
#include "Position.h"
#include "Moves.h"
#include "Player.h"

#include <random>
#include <unordered_set>
//#include <execution>
//#include <type_traits>

class PositionGenerator
{
public:
	PositionGenerator(uint64_t seed = std::random_device{}()) : rnd_engine(seed) {}

	Position Random();
	Position Random(uint64_t empty_count);

	//Position RandomlyPlayed(Position start_pos = Position::Start());
	Position Played(Player&, std::size_t empty_count, Position start = Position::Start());

	//std::unordered_set<Position> RandomlyPlayed(std::size_t count,                       Position start_pos = Position::Start());
	std::vector<Position> Played(Player&, std::size_t count, std::size_t empty_count, Position start = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count,                       Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count, uint64_t empty_count, Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::parallel_policy&& , std::size_t count,                       Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::parallel_policy&& , std::size_t count, uint64_t empty_count, Position start_pos = Position::Start());

	// symmetrically identical positions are considered distinct.
	std::vector<Position> All(std::size_t empty_count, Position start = Position::Start());

	// symmetrically identical positions are considered identical.
	std::vector<Position> AllSymmetricUnique(std::size_t empty_count, Position start = Position::Start());

private:
	std::mt19937_64 rnd_engine;
	
	Board RandomMiddle();
};