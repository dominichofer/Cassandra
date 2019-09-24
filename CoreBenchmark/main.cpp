#include "benchmark/benchmark.h"
#include "Core/PositionGenerator.h"
#include "Core/Search.h"

void EvalGameOver(benchmark::State& state)
{
	PositionGenerator pg;
	const Position pos = pg.Random();

	for (auto _ : state)
		benchmark::DoNotOptimize(EvalGameOver(pg.Random()));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(EvalGameOver);

void PosGen_Random(benchmark::State& state)
{
	PositionGenerator pg;

	for (auto _ : state)
		benchmark::DoNotOptimize(pg.Random());
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PosGen_Random);

void PosGen_Random_empty_count(benchmark::State& state)
{
	PositionGenerator pg;
	const uint64_t empty_count = state.range(0);

	for (auto _ : state)
		benchmark::DoNotOptimize(pg.Random(empty_count));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PosGen_Random_empty_count)->Arg(0)->Arg(20)->Arg(40)->Arg(60);

void PosGen_RandomPlayed(benchmark::State& state)
{
	PositionGenerator pg;
	auto player = RandomPlayer();
	const uint64_t empty_count = state.range(0);

	for (auto _ : state)
		benchmark::DoNotOptimize(pg.Played(player, empty_count));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PosGen_RandomPlayed)->Arg(0)->Arg(20)->Arg(40)->Arg(60);

void PosGen_All(benchmark::State& state)
{
	PositionGenerator pg;
	const uint64_t empty_count = state.range(0);
	std::vector<Position> vec;
	for (auto _ : state)
		pg.All(std::back_inserter(vec), empty_count);
}
BENCHMARK(PosGen_All)->DenseRange(51, 60);

BENCHMARK_MAIN();