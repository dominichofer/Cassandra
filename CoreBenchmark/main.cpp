#include "pch.h"
#include <random>
#include <numeric>

BitBoard RandomBitBoard()
{
	static std::mt19937_64 rng;
	static std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
	return dist(rng);
}

void popcount(benchmark::State& state)
{
	BitBoard b = RandomBitBoard();
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
	BitBoard b = RandomBitBoard();
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
	BitBoard b = RandomBitBoard();
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
	BitBoard b = RandomBitBoard();
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
	BitBoard b = RandomBitBoard();
	for (auto _ : state)
	{
		auto value = FlipHorizontal(b);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipHorizontal);

void FlipToUnique(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
	{
		auto value = FlipToUnique(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(FlipToUnique);

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

void CountLastFlip(benchmark::State& state)
{
	Position pos = RandomPosition();
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

void HasMoves_x64(benchmark::State& state)
{
	Position pos = RandomPosition();
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
	Position pos = RandomPosition();
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
	Position pos = RandomPosition();
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
	Position pos = RandomPosition();
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

void Children(int empty_count)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto generator = Children(Position::Start(), empty_count);
	std::vector<Position> value(generator.begin(), generator.end());
	benchmark::DoNotOptimize(value);
	auto stop = std::chrono::high_resolution_clock::now();

	fmt::print("Children empty_count {:<14}{:>3} ms\n", empty_count, std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void UniqueChildren(int empty_count)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto value = UniqueChildren(Position::Start(), empty_count);
	benchmark::DoNotOptimize(value);
	auto stop = std::chrono::high_resolution_clock::now();

	fmt::print("Unique children empty_count {:<7}{:>3} ms\n", empty_count, std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void RandomGame()
{
	int size = 100'000;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < size; i++)
	{
		Game value = RandomGame(Position::Start());
		benchmark::DoNotOptimize(value);
	}
	auto stop = std::chrono::high_resolution_clock::now();

	fmt::print("RandomGamesFrom {:>22} ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>((stop - start) / size).count());
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
	RandomGame();
	return 0;
}