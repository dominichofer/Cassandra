#include "pch.h"
#include "Board/Board.h"

void EndScore(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
	{
		auto value = EndScore(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(EndScore);

BENCHMARK_MAIN();
