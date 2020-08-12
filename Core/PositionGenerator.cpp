#include "PositionGenerator.h"
#include "Bit.h"

Position PosGen::Random::operator()()
{
	// Each field has a:
	//  25% chance to belong to player,
	//  25% chance to belong to opponent,
	//  50% chance to be empty.

	auto rnd = [this]() { return std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFULL)(rnd_engine); };
	BitBoard a = rnd() & mask;
	BitBoard b = rnd() & mask;
	return { a & ~b, b & ~a };
}

Position PosGen::Random_with_empty_count::operator()()
{
	auto dichotron = [this]() { return std::uniform_int_distribution<int>(0, 1)(rnd_engine) == 0; };

	BitBoard P = 0;
	BitBoard O = 0;
	for (std::size_t e = 64; e > empty_count; e--)
	{
		auto rnd = std::uniform_int_distribution<std::size_t>(0, e - 1)(rnd_engine);
		auto bit = BitBoard(PDep(1ULL << rnd, Position(P, O).Empties()));

		if (dichotron())
			P |= bit;
		else
			O |= bit;
	}
	return { P, O };
}


PosGen::All_after_nth_ply::All_after_nth_ply(std::size_t plies, std::size_t plies_per_pass, Position start)
	: plies(plies), plies_per_pass(plies_per_pass)
{
	stack.push({ start, PossibleMoves(start) });
}

std::optional<Position> PosGen::All_after_nth_ply::operator()()
{
	if (!stack.empty() && (plies == 0))
	{
		auto pos = stack.top().pos;
		stack.pop();
		return pos;
	}

	while (!stack.empty())
	{
		auto& pos = stack.top().pos;
		auto& moves = stack.top().moves;

		assert(stack.size() <= plies);

		if (!moves)
		{
			stack.pop();
			continue;
		}

		Position next = Play(pos, moves.ExtractFirst());
		if (stack.size() == plies)
			return next;

		auto possible_moves = PossibleMoves(next);
		if (!possible_moves)
		{
			for (std::size_t i = 0; i < plies_per_pass; i++)
				stack.push({ next, Moves{0} });
			Position next = PlayPass(next);
			possible_moves = PossibleMoves(next);
			if (stack.size() == plies && !!possible_moves)
				return next;
		}
		stack.push({ next, possible_moves });
	}
	return std::nullopt;
}


PosGen::All_with_empty_count::All_with_empty_count(std::size_t empty_count, Position start)
	: empty_count(empty_count)
{
	if (start.EmptyCount() < empty_count)
		throw;
	stack.push({ start, PossibleMoves(start) });
}

std::optional<Position> PosGen::All_with_empty_count::operator()()
{
	if (!stack.empty() && (stack.top().pos.EmptyCount() == empty_count))
	{
		auto pos = stack.top().pos;
		stack.pop();
		return pos;
	}

	while (!stack.empty())
	{
		auto& pos = stack.top().pos;
		auto& moves = stack.top().moves;

		assert(pos.EmptyCount() > empty_count);

		if (!moves)
		{
			stack.pop();
			continue;
		}

		Position next = Play(pos, moves.ExtractFirst());
		if (next.EmptyCount() == empty_count)
			return next;

		auto possible_moves = PossibleMoves(next);
		if (!possible_moves)
		{ // next has no possible moves. It will be skipped.
			Position next = PlayPass(next);
			possible_moves = PossibleMoves(next);
		}
		stack.push({ next, possible_moves });
	}
	return std::nullopt;
}


PosGen::Played::Played(Player& first, Player& second, std::size_t empty_count, Position start)
	: first(first), second(second), empty_count(empty_count), start(start)
{
	if (start.EmptyCount() < empty_count)
		throw;
}

Position PosGen::Played::operator()()
{
	Position pos = start;
	if (pos.EmptyCount() == empty_count)
		return pos;

	while (true)
	{
		Position old = pos;

		pos = first.Play(pos);
		if (pos.EmptyCount() == empty_count)
			return pos;

		pos = second.Play(pos);
		if (pos.EmptyCount() == empty_count)
			return pos;

		if (old == pos) // both players passed
			pos = start; // restart
	}
}
