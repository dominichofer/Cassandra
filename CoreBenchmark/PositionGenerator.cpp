#include "benchmark/benchmark.h"

#include "Core/PositionGenerator.h"

void PositionGenerator_Random(benchmark::State& state)
{
	PositionGenerator pg;

	for (auto _ : state)
		benchmark::DoNotOptimize(pg.Random());
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PositionGenerator_Random);

void PositionGenerator_Random(benchmark::State& state)
{
	PositionGenerator pg;

	for (auto _ : state)
		benchmark::DoNotOptimize(pg.Random());
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PositionGenerator_Random);

BENCHMARK_MAIN();