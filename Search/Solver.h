#pragma once
#include "Search/Search.h"
#include <cstdint>
#include <chrono>
#include <span>

class Solver
{
	Algorithm& alg;
	bool parallel, silent;
	Table table;
public:
	int64_t abs_err;
	uint64_t nodes;
	std::chrono::duration<double> time;
	int counter;

	Solver(Algorithm&, bool parallel = true, bool silent = false) noexcept;

	void Solve(std::span<const Position>, OpenInterval window, int depth, float confidence_level);
	void Solve(std::span<const Position>);
	void Solve(std::span<const PosScore>, OpenInterval window, int depth, float confidence_level);
	void Solve(std::span<const PosScore>);
	void Clear();

	void PrintHeader() const;
	void PrintSummary() const;
private:
	void PrintRow(int index, const Result& result, int score, std::chrono::duration<double> time, uint64_t nodes);
};
