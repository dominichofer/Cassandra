#include "pch.h"
#include <cstdint>
#include <random>
#include <numeric>
#include <format>
#include <iostream>

uint64_t RandomUint64()
{
	static std::mt19937_64 rng;
	static std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	return dist(rng);
}

void FlippedCodiagonal(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = FlippedCodiagonal(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlippedCodiagonal);

void FlippedDiagonal(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = FlippedDiagonal(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlippedDiagonal);

void FlippedHorizontal(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = FlippedHorizontal(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlippedHorizontal);

void FlippedVertical(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = FlippedVertical(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlippedVertical);

void FlippedToUnique(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
	{
		auto value = FlippedToUnique(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlippedToUnique);

void EmptyCount(benchmark::State& state)
{
	Position pos = RandomPosition();
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
	Position pos = RandomPosition();
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

void PossibleMoves_x64(benchmark::State& state)
{
	Position pos = RandomPosition();
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
	Position pos = RandomPosition();
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
	Position pos = RandomPosition();
	for (auto _ : state)
	{
		auto value = PossibleMoves(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PossibleMoves);

void Children(int empty_count)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto generator = Children(Position::Start(), empty_count);
	std::vector<Position> value(generator.begin(), generator.end());
	benchmark::DoNotOptimize(value);
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << std::format("Children empty_count={:<14}{:>3} ms\n", empty_count, std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void UniqueChildren(int empty_count)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto value = UniqueChildren(Position::Start(), empty_count);
	benchmark::DoNotOptimize(value);
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << std::format("Unique children empty_count={:<7}{:>3} ms\n", empty_count, std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}


int main(int argc, char** argv)
{
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();

	Children(/*empty_count*/ 51);
	UniqueChildren(/*empty_count*/ 51);
	return 0;
}