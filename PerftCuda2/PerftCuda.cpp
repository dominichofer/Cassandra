#include "PerftCuda.h"
#include "Board/Board.h"
#include <cstdint>

// forward declaration
int64_t perft_cuda(const Position&, const int depth, const int cuda_depth);

int64_t CudaHashTablePerft::calculate_n(const Position& pos, const int depth)
{
	if (depth <= 3)
		return UnrolledPerft::calculate(pos, depth);
	if (depth <= pre_cuda_unroll + cuda_depth)
		return perft_cuda(pos, depth, cuda_depth);

	if (auto ret = hash_table.LookUp({ pos, depth }); ret.has_value())
		return ret.value();

	auto moves = PossibleMoves(pos);
	if (!moves)
	{
		Position passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return calculate_n(FlippedToUnique(passed), depth-1);
		return 0;
	}

	int64_t sum = 0;
	for (Field move : moves)
		sum += calculate_n(FlippedToUnique(Play(pos, move)), depth-1);

	hash_table.Update({ pos, depth }, sum);
	return sum;
}

int64_t CudaHashTablePerft::calculate(const Position& pos, const int depth)
{
	if (initial_unroll == 0 || depth - initial_unroll <= 2)
		return calculate_n(pos, depth);

	std::vector<Position> work;
	for (const auto& pos : Children(pos, initial_unroll, true))
		work.push_back(FlippedToUnique(pos));

	int64_t sum = 0;
	int64_t size = static_cast<int64_t>(work.size());
	#pragma omp parallel for schedule(dynamic,1) reduction(+:sum)
	for (int64_t i = 0; i < size; i++)
		sum += calculate_n(work[i], depth - initial_unroll);
	return sum;
}

int64_t CudaHashTablePerft::calculate(const int depth)
{
	if (depth == 0)
		return calculate(Position::Start(), depth);

	// Makes use of 4-fold symmetrie.
	Position pos = Position::Start();
	pos = Play(pos, PossibleMoves(pos).front());
	return 4 * calculate(pos, depth-1);
}
