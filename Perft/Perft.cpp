#include "Perft.h"
#include "PerftCuda/kernel.cuh"
#include <cstdint>
#include <algorithm>
#include <omp.h>

int64_t Correct(int depth)
{
	const int64_t correct[] =
	{
								1,
								4,
							   12,
							   56,
							  244,
							1'396,
							8'200,
						   55'092,
						  390'216,
						3'005'288,
					   24'571'056,
					  212'258'216,
					1'939'879'668,
				   18'429'618'408,
				  184'041'761'768,
				1'891'831'332'208,
			   20'301'171'282'452,
			  222'742'563'853'912,
			2'534'535'926'617'852,
		   29'335'558'770'589'276,
		  349'980'362'625'040'712,
		4'228'388'321'175'157'140,
							   22,
							   23,
							   24,
	};
	return correct[depth];
}


// for 1 ply left
int64_t Perft::calculate_1(const Position& pos) const
{
	auto moves = PossibleMoves(pos);
	if (!moves)
		return PossibleMoves(PlayPass(pos)) ? 1 : 0;
	return moves.size();
}

// for 2 plies left
int64_t Perft::calculate_2(const Position& pos) const
{
	auto moves = PossibleMoves(pos);
	if (!moves)
		return PossibleMoves(PlayPass(pos)).size();

	int64_t sum = 0;
	for (Field move : moves)
		sum += calculate_1(Play(pos, move));
	return sum;
}

int64_t Perft::calculate_n(Position pos, const int depth)
{
	switch (depth)
	{
		case 0: return 1;
		case 1: return calculate_1(pos);
		case 2: return calculate_2(pos);
		default:
			if (cuda and depth <= cuda_depth)
				return perft_cuda(pos, depth);
			break;
	}

	pos = FlippedToUnique(pos);
	auto moves = PossibleMoves(pos);
	if (!moves)
	{
		Position passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return calculate_n(passed, depth - 1);
		return 0;
	}


	if (uint64_t ret = tt.LookUp(pos, depth))
		return ret;

	int64_t sum = 0;
	for (Field move : moves)
		sum += calculate_n(Play(pos, move), depth - 1);

	tt.Insert(pos, depth, sum);
	return sum;
}

int64_t Perft::calculate(const Position& pos, const int depth)
{
	if (initial_unroll == 0 or depth <= initial_unroll + 2)
		return calculate_n(pos, depth);

	std::vector<Position> work(std::from_range, Children(pos, initial_unroll, true));

	int64_t sum = 0;
	#pragma omp parallel for schedule(static,1) reduction(+:sum)
	for (int64_t i = 0; i < static_cast<int64_t>(work.size()); i++)
		sum += calculate_n(work[i], depth - initial_unroll);
	return sum;
}

int64_t Perft::calculate(const int depth)
{
	if (depth == 0)
		return calculate(Position::Start(), depth);

	// Use 4-fold symmetry.
	Position pos = Position::Start();
	pos = Play(pos, PossibleMoves(pos).front());
	return 4 * calculate(pos, depth - 1);
}
