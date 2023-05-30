#include "benchmark/benchmark.h"
#include "Pattern/Pattern.h"

void PatternEvalH(benchmark::State& state)
{
	auto evaluator = ScoreEstimator(pattern::L0);
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
	auto evaluator = ScoreEstimator(pattern::D7);
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
	auto evaluator = ScoreEstimator(pattern::B5);
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
	auto evaluator = ScoreEstimator({ pattern::L0, pattern::D7, pattern::B5 });
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
	auto evaluator = ScoreEstimator(pattern::logistello);
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
	auto evaluator = ScoreEstimator(pattern::edax);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEdax);

void MSSE_Edax(benchmark::State& state)
{
	auto evaluator = MSSE(/*stage_size*/ 5, pattern::edax);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(MSSE_Edax);

BENCHMARK_MAIN();