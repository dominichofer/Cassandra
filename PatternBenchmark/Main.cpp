#include "benchmark/benchmark.h"
#include "Pattern/Pattern.h"

void PatternEvalH(benchmark::State& state)
{
	auto evaluator = GLEM(pattern::L0);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalH);

void PatternEvalD(benchmark::State& state)
{
	auto evaluator = GLEM(pattern::D7);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalD);

void PatternEvalA(benchmark::State& state)
{
	auto evaluator = GLEM(pattern::B5);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalA);

void PatternEvalHDA(benchmark::State& state)
{
	auto evaluator = GLEM({ pattern::L0, pattern::D7, pattern::B5 });
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalHDA);

void PatternLogistello(benchmark::State& state)
{
	auto evaluator = GLEM(pattern::logistello);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternLogistello);

void PatternEdax(benchmark::State& state)
{
	auto evaluator = GLEM(pattern::edax);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEdax);

void AAGLEM_Edax(benchmark::State& state)
{
	auto evaluator = AAGLEM(pattern::edax, /*block_size*/ 10);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(AAGLEM_Edax);

BENCHMARK_MAIN();