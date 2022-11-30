#include "PositionGenerator.h"
#include "Bit.h"

Position RandomPositionGenerator::operator()() noexcept
{
	// Each field has a:
	//  25% chance to belong to player,
	//  25% chance to belong to opponent,
	//  50% chance to be empty.

	BitBoard a{ dist(rnd_engine) };
	BitBoard b{ dist(rnd_engine) };
	return { a & ~b, b & ~a };
}

RandomPositionGeneratorWithEmptyCount::RandomPositionGeneratorWithEmptyCount(int empty_count, unsigned int seed) noexcept(false)
	: empty_count(empty_count), rnd_engine(seed)
{
	if (empty_count < 0 or empty_count > 64)
		throw std::runtime_error("'empty_count' out of bounds.");
}

Position RandomPositionGeneratorWithEmptyCount::operator()() noexcept
{
	Position pos;
	while (pos.EmptyCount() > empty_count)
	{
		int rnd = std::uniform_int_distribution<int>(0, pos.EmptyCount() - 1)(rnd_engine);

		// deposit bit on an empty field
		auto bit = BitBoard(PDep(1ULL << rnd, pos.Empties()));

		if (boolean(rnd_engine))
			pos = Position{ pos.Player() | bit, pos.Opponent() };
		else
			pos = Position{ pos.Player(), pos.Opponent() | bit };
	}
	return pos;
}

Position RandomPosition(unsigned int seed)
{
	return RandomPositionGenerator{ seed }();
}

std::vector<Position> RandomPositions(int count, unsigned int seed)
{
	RandomPositionGenerator gen{ seed };
	std::vector<Position> ret;
	std::generate_n(std::back_inserter(ret), count, gen);
	return ret;
}

Position RandomPositionWithEmptyCount(int empty_count, unsigned int seed) noexcept(false)
{
	return RandomPositionGeneratorWithEmptyCount{ empty_count, seed }();
}

std::vector<Position> RandomPositionsWithEmptyCount(int count, int empty_count, unsigned int seed) noexcept(false)
{
	RandomPositionGeneratorWithEmptyCount gen{ empty_count, seed };
	std::vector<Position> ret;
	std::generate_n(std::back_inserter(ret), count, gen);
	return ret;
}

Position RandomlyPlayedPositionWithEmptyCount(int empty_count, Position start, unsigned int seed) noexcept(false)
{
	if (start.EmptyCount() < empty_count)
		throw std::runtime_error("Input has not enough empty fields.");

	RandomPlayer player{ seed };

	Position pos = start;
	int pass_count = 0;
	while (pos.EmptyCount() > empty_count)
	{
		Field move = player.ChooseMove(pos);
		pos = PlayOrPass(pos, move);
		
		if (move == Field::invalid)
			pass_count++;
		else
			pass_count = 0;

		if (pass_count == 2)
			pos = start; // restart
	}
	return pos;
}
Position RandomlyPlayedPositionWithEmptyCount(int empty_count, unsigned int seed) noexcept(false)
{
	return RandomlyPlayedPositionWithEmptyCount(empty_count, Position::Start(), seed);
}

std::vector<Position> RandomlyPlayedPositionsWithEmptyCount(int count, int empty_count, Position start, unsigned int seed) noexcept(false)
{
	std::vector<Position> ret(count, start);
	#pragma omp parallel for
	for (int i = 0; i < count; i++)
		ret[i] = RandomlyPlayedPositionWithEmptyCount(empty_count, start, seed + i);
	return ret;
}
std::vector<Position> RandomlyPlayedPositionsWithEmptyCount(int count, int empty_count, unsigned int seed) noexcept(false)
{
	return RandomlyPlayedPositionsWithEmptyCount(count, empty_count, Position::Start(), seed);
}
