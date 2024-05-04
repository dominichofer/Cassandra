#pragma once
#include "Search/Search.h"
#include <cstdint>
#include <chrono>
#include <span>
#include <thread>

class Solver
{
	Algorithm& alg;
	bool silent;
	int threads;
	Table table;
	int64_t abs_err;
	int counter;
public:
	uint64_t nodes;
	std::chrono::duration<double> time;

	Solver(Algorithm&, bool silent = false, int threads = std::thread::hardware_concurrency()) noexcept;

	std::vector<Score> Solve(std::span<const Position>, OpenInterval window, Intensity);
	std::vector<Score> Solve(std::span<const Position>);
	std::vector<Score> Solve(std::span<const ScoredPosition>, OpenInterval window, Intensity);
	std::vector<Score> Solve(std::span<const ScoredPosition>);
	void Clear();
	void PrintHeader() const;
	void PrintSummary() const;
private:
	void PrintRow(int index, const Result&, Score, std::chrono::duration<double> time, uint64_t nodes);
};
