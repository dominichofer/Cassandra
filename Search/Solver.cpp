#include "Solver.h"

Solver::Solver(Algorithm& alg, bool parallel, bool silent) noexcept
	: alg(alg)
	, parallel(parallel)
	, silent(silent)
	, table(
		"        #| depth | eval|score|     time [s] |      nodes     |     N/s     | PV ",
		"{:>9L}|{:7}| {:3} | {:3} | {:>12} |{:>15L} |{:>12L} | {:<}")
{
	Clear();
}

void Solver::Solve(std::span<const Position> pos, OpenInterval window, int depth, float confidence_level)
{
	#pragma omp parallel for schedule(static, 1) if (parallel)
	for (int i = 0; i < pos.size(); i++)
	{
		alg.Nodes() = 0;
		auto start = std::chrono::high_resolution_clock::now();
		auto result = alg.Eval(pos[i], window, depth, confidence_level);
		auto time = std::chrono::high_resolution_clock::now() - start;

		#pragma omp critical
		{
			this->nodes += alg.Nodes();
			this->time += time;
			this->counter++;
			if (not silent)
				PrintRow(i, result, result.score, time, alg.Nodes());
		}
	}
}

void Solver::Solve(std::span<const Position> pos)
{
	Solve(pos, { -inf_score, +inf_score }, 64, inf);
}

void Solver::Solve(std::span<const PosScore> pos, OpenInterval window, int depth, float confidence_level)
{
	#pragma omp parallel for schedule(static, 1) if (parallel)
	for (int i = 0; i < pos.size(); i++)
	{
		alg.Nodes() = 0;
		auto start = std::chrono::high_resolution_clock::now();
		auto result = alg.Eval(pos[i].pos, window, depth, confidence_level);
		auto time = std::chrono::high_resolution_clock::now() - start;

		#pragma omp critical
		{
			this->abs_err += std::abs(result.score - pos[i].score);
			this->nodes += alg.Nodes();
			this->time += time;
			this->counter++;
			if (not silent)
				PrintRow(i, result, pos[i].score, time, alg.Nodes());
		}
	}
}

void Solver::Solve(std::span<const PosScore> pos)
{
	Solve(pos, { -inf_score, +inf_score }, 64, inf);
}

void Solver::Clear()
{
	alg.Clear();
	abs_err = 0;
	nodes = 0;
	time = std::chrono::duration<double>{ 0 };
	counter = 0;
}

void Solver::PrintHeader() const
{
	if (not silent)
		table.PrintHeader();
}

void Solver::PrintSummary() const
{
	if (silent)
		return;

	table.PrintSeparator();
	double avg_abs_err = static_cast<double>(abs_err) / static_cast<double>(counter);
	double nps = static_cast<double>(nodes) / time.count();

	if (avg_abs_err == 0)
		std::cout << std::format("                               {:>12}  {:>15L}  {:>10.0Lf}\n", HH_MM_SS(time), nodes, nps);
	else
		std::cout << std::format("           avg abs err: {:4.1f}   {:>12}  {:>15L}  {:>10.0Lf}\n", avg_abs_err, HH_MM_SS(time), nodes, nps);
}

void Solver::PrintRow(int index, const Result& result, int score, std::chrono::duration<double> time, uint64_t nodes)
{
	std::string depth = DepthClToString(result.depth, result.confidence_level);
	std::string eval = ScoreToString(result.score);
	std::string score_str = (result.score == score) ? "" : ScoreToString(score);
	std::string time_str = HH_MM_SS(time);
	double nps = static_cast<double>(nodes) / time.count();
	if (nps == std::numeric_limits<double>::infinity())
		nps = 0;
	std::string pv = to_string(result.best_move);

	table.PrintRow(index, depth, eval, score_str, time_str, nodes, static_cast<uint64_t>(nps), pv);
}
