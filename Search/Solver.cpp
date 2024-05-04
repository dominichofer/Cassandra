#include "Solver.h"

Solver::Solver(Algorithm& alg, bool silent, int threads) noexcept
	: alg(alg)
	, silent(silent)
	, threads(threads)
	, table(
		"        #| depth | eval|score|     time [s] |      nodes     |     N/s     | PV ",
		"{:>9L}|{:7}| {:3} | {:3} | {:>12} |{:>15L} |{:>12L} | {:<}")
{
	Clear();
}

std::vector<Score> Solver::Solve(std::span<const Position> pos, OpenInterval window, Intensity intensity)
{
	Clear();
	if (not silent)
		PrintHeader();

	std::vector<Score> results(pos.size());
	#pragma omp parallel for num_threads(threads) if(threads>1)
	for (int i = 0; i < pos.size(); i++)
	{
		alg.Nodes() = 0;
		auto start = std::chrono::high_resolution_clock::now();
		auto result = alg.Eval(pos[i], window, intensity);
		auto time = std::chrono::high_resolution_clock::now() - start;
		results[i] = result.GetScore();

		#pragma omp critical
		{
			this->nodes += alg.Nodes();
			this->time += time;
			this->counter++;
			if (not silent)
				PrintRow(i, result, result.GetScore(), time, alg.Nodes());
		}
	}
	PrintSummary();
	return results;
}

std::vector<Score> Solver::Solve(std::span<const Position> pos)
{
	return Solve(pos, { min_score, max_score }, { 64 });
}

std::vector<Score> Solver::Solve(std::span<const ScoredPosition> pos, OpenInterval window, Intensity intensity)
{
	Clear();
	if (not silent)
		PrintHeader();

	std::vector<Score> results(pos.size());
	#pragma omp parallel for num_threads(threads) if(threads>1)
	for (int i = 0; i < pos.size(); i++)
	{
		alg.Nodes() = 0;
		auto start = std::chrono::high_resolution_clock::now();
		Intensity limitted_intensity{ std::min<int8_t>(intensity.depth, pos[i].pos.EmptyCount()), intensity.level };
		auto result = alg.Eval(pos[i].pos, window, limitted_intensity);
		auto time = std::chrono::high_resolution_clock::now() - start;
		results[i] = result.GetScore();

		#pragma omp critical
		{
			this->abs_err += std::abs(result.GetScore() - pos[i].score);
			this->nodes += alg.Nodes();
			this->time += time;
			this->counter++;
			if (not silent)
				PrintRow(i, result, pos[i].score, time, alg.Nodes());
		}
	}
	PrintSummary();
	return results;
}

std::vector<Score> Solver::Solve(std::span<const ScoredPosition> pos)
{
	return Solve(pos, { min_score, max_score }, { 64 });
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
	table.PrintHeader();
}

void Solver::PrintSummary() const
{
	if (not silent)
		table.PrintSeparator();
	double avg_abs_err = static_cast<double>(abs_err) / static_cast<double>(counter);
	double nps = static_cast<double>(nodes) / time.count();

	if (avg_abs_err == 0)
		std::cout << std::format("                               {:>12}  {:>15L}  {:>10.0Lf}\n", HH_MM_SS(time), nodes, nps);
	else
		std::cout << std::format("           avg abs err: {:4.1f}   {:>12}  {:>15L}  {:>10.0Lf}\n", avg_abs_err, HH_MM_SS(time), nodes, nps);
}

void Solver::PrintRow(int index, const Result& result, Score score, std::chrono::duration<double> time, uint64_t nodes)
{
	std::string intensity = to_string(result.intensity);
	std::string eval = to_string(result.GetScore());
	std::string score_str = (result.GetScore() == score) ? "" : to_string(score);
	std::string time_str = HH_MM_SS(time);
	double nps = static_cast<double>(nodes) / time.count();
	if (nps == std::numeric_limits<double>::infinity())
		nps = 0;
	std::string pv = to_string(result.best_move);

	table.PrintRow(index, intensity, eval, score_str, time_str, nodes, static_cast<uint64_t>(nps), pv);
}
