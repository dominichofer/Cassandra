#include "benchmark/benchmark.h"
#include "Core/BitBoard.h"
#include "Core/PositionGenerator.h"
#include "Core/Search.h"

void FlipCodiagonal(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const BitBoard b{ dist(rng) };

	for (auto _ : state)
		benchmark::DoNotOptimize(FlipCodiagonal(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipCodiagonal);

void FlipDiagonal(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const BitBoard b{ dist(rng) };

	for (auto _ : state)
		benchmark::DoNotOptimize(FlipDiagonal(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipDiagonal);

void FlipHorizontal(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const BitBoard b{ dist(rng) };

	for (auto _ : state)
		benchmark::DoNotOptimize(FlipHorizontal(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipHorizontal);

void FlipVertical(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const BitBoard b{ dist(rng) };

	for (auto _ : state)
		benchmark::DoNotOptimize(FlipVertical(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipVertical);

void EvalGameOver(benchmark::State& state)
{
	PosGen::Random rnd;
	const Position pos = rnd();

	for (auto _ : state)
		benchmark::DoNotOptimize(EvalGameOver(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(EvalGameOver);

void PosGen_Random(benchmark::State& state)
{
	PosGen::Random rnd;

	for (auto _ : state)
		benchmark::DoNotOptimize(rnd());
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PosGen_Random);

void PosGen_Random_empty_count(benchmark::State& state)
{
	const uint64_t empty_count = state.range(0);
	PosGen::Random_with_empty_count rnd(empty_count);

	for (auto _ : state)
		benchmark::DoNotOptimize(rnd());
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PosGen_Random_empty_count)->Arg(0)->Arg(20)->Arg(40)->Arg(60);

void PosGen_RandomPlayed(benchmark::State& state)
{
	auto player1 = RandomPlayer();
	auto player2 = RandomPlayer();
	const uint64_t empty_count = state.range(0);
	PosGen::Played generate(player1, player2, empty_count);

	for (auto _ : state)
		benchmark::DoNotOptimize(generate());
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PosGen_RandomPlayed)->Arg(0)->Arg(20)->Arg(40)->Arg(60);

void PosGen_All(benchmark::State& state)
{
	const uint64_t empty_count = state.range(0);
	PosGen::All_with_empty_count rnd(empty_count);
	for (auto _ : state)
	{
		std::vector<Position> vec;
		generate_all(std::back_inserter(vec), rnd);
	}
}
BENCHMARK(PosGen_All)->DenseRange(51, 60);

BENCHMARK_MAIN();