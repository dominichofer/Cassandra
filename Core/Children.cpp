#include "Children.h"

using namespace children;

Iterator::Iterator(const Position& start, int plies, bool pass_is_a_ply) noexcept
	: plies(plies), pass_is_a_ply(pass_is_a_ply)
{
	stack.reserve(plies);

	if (stack.size() == plies) {
		stack.emplace_back(start, Moves{});
		return;
	}

	const auto moves = PossibleMoves(start);
	if (moves)
		stack.emplace_back(start, moves);
	else
	{
		const auto passed = PlayPass(start);
		const auto passed_moves = PossibleMoves(passed);
		if (passed_moves)
		{
			if (pass_is_a_ply)
			{
				stack.emplace_back(start, Moves{});
				if (stack.size() == plies + 1)
					return;
			}
			stack.emplace_back(passed, passed_moves);
		}
	}
	if (stack.size() == plies + 1)
		return;
	++(*this);
}

Iterator& Iterator::operator++()
{
	while (not stack.empty())
	{
		if (stack.size() == plies + 1 || !stack.back().moves) {
			stack.pop_back();
			continue;
		}

		const auto move = stack.back().moves.front();
		stack.back().moves.pop_front();
		const auto pos = Play(stack.back().pos, move);

		if (stack.size() == plies) {
			stack.emplace_back(pos, Moves{});
			return *this;
		}

		const auto moves = PossibleMoves(pos);
		if (moves)
			stack.emplace_back(pos, moves);
		else
		{
			const auto passed = PlayPass(pos);
			const auto passed_moves = PossibleMoves(passed);
			if (passed_moves)
			{
				if (pass_is_a_ply)
				{
					stack.emplace_back(pos, Moves{});
					if (stack.size() == plies + 1)
						return *this;
				}
				stack.emplace_back(passed, passed_moves);
			}
		}
		if (stack.size() == plies + 1)
			return *this;
	}
	return *this;
}

children::Generator Children(Position start, int plies, bool pass_is_a_ply)
{
	assert(plies >= 0);
	return { start, plies, pass_is_a_ply };
}

children::Generator Children(Position start, int empty_count)
{
	assert(start.EmptyCount() >= empty_count);
	return { start, start.EmptyCount() - empty_count, false };
}

std::set<Position> UniqueChildren(Position start, int empty_count)
{
	std::set<Position> ret;
	for (Position pos : Children(start, empty_count))
		ret.insert(FlipToUnique(pos));
	return ret;
}
