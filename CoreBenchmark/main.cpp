#include "pch.h"
#include <random>
#include <numeric>

void popcount(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const uint64_t b = dist(rng);

	for (auto _ : state)
	{
		auto value = popcount(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(popcount);


void FlipVertical(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const BitBoard b{ dist(rng) };

	for (auto _ : state)
	{
		auto value = FlipVertical(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipVertical);

void FlipCodiagonal(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const BitBoard b{ dist(rng) };

	for (auto _ : state)
	{
		auto value = FlipCodiagonal(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipCodiagonal);

void FlipDiagonal(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	const BitBoard b{ dist(rng) };

	for (auto _ : state)
	{
		auto value = FlipDiagonal(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipDiagonal);

void FlipHorizontal(benchmark::State& state)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	BitBoard b{ dist(rng) };

	for (auto _ : state)
	{
		auto value = FlipHorizontal(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipHorizontal);

void EmptyCount(benchmark::State& state)
{
	auto pos = PosGen::Random{}();

	for (auto _ : state)
	{
		auto value = pos.EmptyCount();
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(EmptyCount);

void Flips(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	unsigned int move = 0;

	for (auto _ : state)
	{
		move = (move + 1) % 64;
		auto value = Flips(pos, static_cast<Field>(move));
		benchmark::DoNotOptimize(value);
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
		auto value = CountLastFlip(pos, static_cast<Field>(move));
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(CountLastFlip);

void PossibleMoves_x64(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
	{
		auto value = detail::PossibleMoves_x64(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves_x64);

void PossibleMoves_AVX2(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
	{
		auto value = detail::PossibleMoves_AVX2(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves_AVX2);

void PossibleMoves(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
	{
		auto value = PossibleMoves(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves);

void HasMoves_x64(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
	{
		auto value = detail::HasMoves_x64(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(HasMoves_x64);

void HasMoves_AVX2(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
	{
		auto value = detail::HasMoves_AVX2(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(HasMoves_AVX2);

void HasMoves(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
	{
		auto value = HasMoves(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(HasMoves);

void EvalGameOver(benchmark::State& state)
{
	auto pos = PosGen::Random{}();
	for (auto _ : state)
	{
		auto value = EvalGameOver(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(EvalGameOver);

//void StableEdges(benchmark::State& state)
//{
//	auto pos = PosGen::Random{}();
//	for (auto _ : state)
//	{
//		auto value = StableEdges(pos);
//		benchmark::DoNotOptimize(value);
//	}
//	state.SetItemsProcessed(state.iterations());
//}
//BENCHMARK(StableEdges);
//
//void StableStonesOpponent(benchmark::State& state)
//{
//	auto pos = PosGen::Random{}();
//	for (auto _ : state)
//	{
//		auto value = StableStonesOpponent(pos);
//		benchmark::DoNotOptimize(value);
//	}
//	state.SetItemsProcessed(state.iterations());
//}
//BENCHMARK(StableStonesOpponent);
//


BENCHMARK_MAIN();