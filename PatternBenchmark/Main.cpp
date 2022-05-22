#include "benchmark/benchmark.h"
#include "Pattern/Pattern.h"

const std::vector<BitBoard> edax_pattern =
{
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - # # #"
		"- - - - - # # #"
		"- - - - - # # #"_BitBoard,

		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - # # # # #"_BitBoard,

		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- # - - - - # -"
		"# # # # # # # #"_BitBoard,

		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - # # # # - -"
		"# - # # # # - #"_BitBoard,

		BitBoard::HorizontalLine(1), // L1
		BitBoard::HorizontalLine(2), // L2
		BitBoard::HorizontalLine(3), // L3
		BitBoard::CodiagonalLine(0), // D8
		BitBoard::CodiagonalLine(1), // D7
		BitBoard::CodiagonalLine(2), // D6
		BitBoard::CodiagonalLine(3), // D5
		BitBoard::CodiagonalLine(4), // D4
};

void PatternEvalH(benchmark::State& state)
{
	auto evaluator = GLEM(BitBoard::HorizontalLine(1));
	auto pos = PosGen::Random{}();

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
	auto evaluator = GLEM(BitBoard::CodiagonalLine(1));
	auto pos = PosGen::Random{}();

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
	auto evaluator = GLEM(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # # # # #"
		"- - - # # # # #"_BitBoard);
	auto pos = PosGen::Random{}();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalA);

void PatternEdax(benchmark::State& state)
{
	auto evaluator = GLEM(edax_pattern);
	auto pos = PosGen::Random{}();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEdax);

void AA_GLEM(benchmark::State& state)
{
	auto evaluator = AAGLEM(edax_pattern, { 0, 10, 20, 30, 40, 50, 64 });
	auto pos = PosGen::Random{}();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(AA_GLEM);

BENCHMARK_MAIN();