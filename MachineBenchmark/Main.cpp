#include "benchmark/benchmark.h"

#include "Machine/BitTwiddling.h"
#include "Machine/CountLastFlip.h"
#include "Machine/Flips.h"
#include "Machine/PossibleMoves.h"
#include "Machine/Stability.h"

#include <random>

using namespace detail;

void BitScanLSB(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(BitScanLSB(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BitScanLSB);

void BitScanMSB(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(BitScanMSB(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BitScanMSB);

void CountLeadingZeros(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(CountLeadingZeros(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(CountLeadingZeros);

void CountTrailingZeros(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(CountTrailingZeros(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(CountTrailingZeros);

void GetLSB(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(GetLSB(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(GetLSB);

void GetMSB(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(GetMSB(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(GetMSB);

void RemoveLSB_generic(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	uint64_t b = dist(rng);

	for (auto _ : state)
	{
		benchmark::DoNotOptimize(b);
		detail::RemoveLSB_generic(b);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(RemoveLSB_generic);

void RemoveLSB_intrinsic(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	uint64_t b = dist(rng);

	for (auto _ : state)
	{
		benchmark::DoNotOptimize(b);
		detail::RemoveLSB_intrinsic(b);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(RemoveLSB_intrinsic);

void RemoveMSB(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	uint64_t b = dist(rng);

	for (auto _ : state)
	{
		benchmark::DoNotOptimize(b);
		detail::RemoveLSB_intrinsic(b);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(RemoveMSB);

void PopCount_generic(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(detail::PopCount_generic(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PopCount_generic);

void PopCount_intrinsic(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFui64 };
	const uint64_t b = dist(rng);

	for (auto _ : state)
		benchmark::DoNotOptimize(detail::PopCount_intrinsic(b));
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PopCount_intrinsic);


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

void StableEdges(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFui64);
	uint64_t P = dist(rng);
	uint64_t O = dist(rng);

	for (auto _ : state)
	{
		P = P * 16807 + 1;
		O = O * 48271 + 3;
		benchmark::DoNotOptimize(StableEdges(Position(BitBoard(P), BitBoard(O))));
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(StableEdges);

void StableStones(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFui64);
	uint64_t P = dist(rng);
	uint64_t O = dist(rng);

	for (auto _ : state)
	{
		P = P * 16807 + 1;
		O = O * 48271 + 3;
		benchmark::DoNotOptimize(StableStones(Position(BitBoard(P), BitBoard(O))));
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(StableStones);

BENCHMARK_MAIN();
