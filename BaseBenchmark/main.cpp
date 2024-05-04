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

void popcount(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = std::popcount(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(popcount);

void GetLSB(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = ::GetLSB(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(GetLSB);

void ClearLSB(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		::ClearLSB(b);
		benchmark::DoNotOptimize(b);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(ClearLSB);

void BExtr(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = ::BExtr(b, 3, 7);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BExtr);

void PDep(benchmark::State& state)
{
	uint64_t a = RandomUint64();
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = ::PDep(a, b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PDep);

void PExt(benchmark::State& state)
{
	uint64_t a = RandomUint64();
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = ::PExt(a, b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PExt);

void BSwap(benchmark::State& state)
{
	uint64_t b = RandomUint64();
	for (auto _ : state)
	{
		auto value = ::BSwap(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BSwap);

int main(int argc, char** argv)
{
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();

	return 0;
}