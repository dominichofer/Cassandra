#include "benchmark/benchmark.h"

#include "Machine/BitTwiddling.h"
#include "Machine/CountLastFlip.h"
#include "Machine/Flips.h"
#include "Machine/PossibleMoves.h"

#include <random>

void PossibleMoves_x64(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t p = dist(rng);
	const uint64_t o = dist(rng);
	const uint64_t P = (p & ~o);
	const uint64_t O = (o & ~p);

	for (auto _ : state)
		benchmark::DoNotOptimize(PossibleMoves_x64(P, O));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves_x64);

void PossibleMoves_SSE2(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t p = dist(rng);
	const uint64_t o = dist(rng);
	const uint64_t P = (p & ~o);
	const uint64_t O = (o & ~p);

	for (auto _ : state)
		benchmark::DoNotOptimize(PossibleMoves_SSE2(P, O));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves_SSE2);

void PossibleMoves_AVX2(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t p = dist(rng);
	const uint64_t o = dist(rng);
	const uint64_t P = (p & ~o);
	const uint64_t O = (o & ~p);

	for (auto _ : state)
		benchmark::DoNotOptimize(PossibleMoves_AVX2(P, O));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves_AVX2);

void PossibleMoves(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t p = dist(rng);
	const uint64_t o = dist(rng);
	const uint64_t P = (p & ~o);
	const uint64_t O = (o & ~p);

	for (auto _ : state)
		benchmark::DoNotOptimize(PossibleMoves(P, O));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves);


void Flips(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFui64);
	uint64_t P = dist(rng);
	uint64_t O = dist(rng);
	uint64_t move = 0;

	for (auto _ : state)
	{
		P = P * 16807 + 1;
		O = O * 48271 + 3;
		move = (move + 7) & 0x3F;
		benchmark::DoNotOptimize(Flips(P, O, move));
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(Flips);

void CountLastFlip(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFui64);
	uint64_t P = dist(rng);
	uint64_t move = 0;

	for (auto _ : state)
	{
		P = P * 16807 + 1;
		move = (move + 7) & 0x3F;
		benchmark::DoNotOptimize(CountLastFlip(P, move));
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(CountLastFlip);

BENCHMARK_MAIN();
