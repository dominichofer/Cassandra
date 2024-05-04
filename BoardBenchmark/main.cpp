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
		benchmark::DoNotOptimize(FlippedCodiagonal(b));
}
BENCHMARK(FlippedCodiagonal);

void FlippedDiagonal(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
		benchmark::DoNotOptimize(FlippedDiagonal(b));
}
BENCHMARK(FlippedDiagonal);

void FlippedHorizontal(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
		benchmark::DoNotOptimize(FlippedHorizontal(b));
}
BENCHMARK(FlippedHorizontal);

void FlippedVertical(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
		benchmark::DoNotOptimize(FlippedVertical(b));
}
BENCHMARK(FlippedVertical);

void FlippedToUnique(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(FlippedToUnique(pos));
}
BENCHMARK(FlippedToUnique);

void EmptyCount(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(pos.EmptyCount());
}
BENCHMARK(EmptyCount);

void CountLastFlip(benchmark::State& state)
{
	Position pos = RandomPosition();
	unsigned int move = 0;
	for (auto _ : state)
		benchmark::DoNotOptimize(CountLastFlip(pos, static_cast<Field>(move++ % 64)));
}
BENCHMARK(CountLastFlip);

void Flips_x64(benchmark::State& state)
{
	Position pos = RandomPosition();
	unsigned int move = 0;
	for (auto _ : state)
		benchmark::DoNotOptimize(detail::Flips_x64(pos, static_cast<Field>(move++ % 64)));
}
BENCHMARK(Flips_x64);

void Flips_AVX2(benchmark::State& state)
{
	Position pos = RandomPosition();
	unsigned int move = 0;
	for (auto _ : state)
		benchmark::DoNotOptimize(detail::Flips_AVX2(pos, static_cast<Field>(move++ % 64)));
}
BENCHMARK(Flips_AVX2);

void Flips(benchmark::State& state)
{
	Position pos = RandomPosition();
	unsigned int move = 0;
	for (auto _ : state)
		benchmark::DoNotOptimize(Flips(pos, static_cast<Field>(move++ % 64)));
}
BENCHMARK(Flips);

void PossibleMoves_x64(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(detail::PossibleMoves_x64(pos));
}
BENCHMARK(PossibleMoves_x64);

void PossibleMoves_AVX2(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(detail::PossibleMoves_AVX2(pos));
}
BENCHMARK(PossibleMoves_AVX2);

void PossibleMoves(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(PossibleMoves(pos));
}
BENCHMARK(PossibleMoves);

void StableEdges(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(StableEdges(pos));
}
BENCHMARK(StableEdges);

void StableStonesOpponent(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(StableStonesOpponent(pos));
}
BENCHMARK(StableStonesOpponent);

void Children(int empty_count)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto generator = Children(Position::Start(), empty_count);
	std::vector<Position> value(generator.begin(), generator.end());
	benchmark::DoNotOptimize(value);
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << std::format("Children( empty_count={} ) {:>3} ms\n", empty_count, std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void UniqueChildren(int empty_count)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto value = UniqueChildren(Position::Start(), empty_count);
	benchmark::DoNotOptimize(value);
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << std::format("UniqueChildren( empty_count={} ) {:>3} ms\n", empty_count, std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}


int main(int argc, char** argv)
{
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();

	Children(/*empty_count*/ 50);
	UniqueChildren(/*empty_count*/ 51);
	return 0;
}