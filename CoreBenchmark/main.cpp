#include "benchmark/benchmark.h"
#include "Core/Core.h"
#include <random>
#include "Pattern/Evaluator.h"
#include "Pattern/DenseIndexer.h"

using namespace detail;
using namespace Pattern;

constexpr BitBoard PatternH{ 0x00000000000000E7ULL }; // HorizontalSymmetric
constexpr BitBoard PatternD{ 0x8040201008040303ULL }; // DiagonalSymmetric
constexpr BitBoard PatternA{ 0x000000000000000FULL }; // Asymmetric
static const auto WeightsH = Weights(CreateDenseIndexer(PatternH)->reduced_size, 0);
static const auto WeightsD = Weights(CreateDenseIndexer(PatternD)->reduced_size, 0);
static const auto WeightsA = Weights(CreateDenseIndexer(PatternA)->reduced_size, 0);

void PatternEvalH(benchmark::State& state)
{
	auto indexer = CreateDenseIndexer(PatternH);

	Weights compressed(indexer->reduced_size);
	std::iota(compressed.begin(), compressed.end(), 1);
	auto evaluator = CreateEvaluator(PatternH, compressed);
	auto gen = PosGen::Random(13);

	for (auto _ : state)
		benchmark::DoNotOptimize(evaluator->Eval(gen()));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalH);

void PatternEvalA(benchmark::State& state)
{
	auto indexer = CreateDenseIndexer(PatternA);

	Weights compressed(indexer->reduced_size);
	std::iota(compressed.begin(), compressed.end(), 1);
	auto evaluator = CreateEvaluator(PatternA, compressed);
	auto gen = PosGen::Random(13);

	for (auto _ : state)
		benchmark::DoNotOptimize(evaluator->Eval(gen()));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalA);

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

void popcount(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(popcount(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(popcount);

void HasMoves_x64(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(HasMoves_x64(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(HasMoves_x64);

void HasMoves_AVX2(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(HasMoves_AVX2(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(HasMoves_AVX2);

void HasMoves(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(HasMoves(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(HasMoves);

void PossibleMoves_x64(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(detail::PossibleMoves_x64(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves_x64);

void PossibleMoves_AVX2(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(detail::PossibleMoves_AVX2(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves_AVX2);

void PossibleMoves(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(PossibleMoves(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves);

void Flips(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	unsigned int move = 0;

	for (auto _ : state)
	{
		move = (move + 1) % 64;
		benchmark::DoNotOptimize(Flips(pos, static_cast<Field>(move)));
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(Flips);

void CountLastFlip(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	unsigned int move = 0;

	for (auto _ : state)
	{
		move = (move + 1) % 64;
		benchmark::DoNotOptimize(CountLastFlip(pos, static_cast<Field>(move)));
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(CountLastFlip);

void StableEdges(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(StableEdges(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(StableEdges);

void StableStones(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
		benchmark::DoNotOptimize(StableStones(pos));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(StableStones);

void EvalGameOver(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
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

BENCHMARK_MAIN();