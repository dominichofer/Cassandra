#include "benchmark/benchmark.h"
#include "Core/Core.h"
#include <random>

using namespace detail;

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

BENCHMARK_MAIN();