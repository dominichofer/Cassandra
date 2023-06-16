#include "ResultTable.h"
#include "String.h"
#include <cstdint>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>

ResultTable::ResultTable() : Table(
	"        #| depth| eval|score|     time [s] |      nodes     |     N/s     | PV",
	"{:>9L}|{:6}| {:3} | {:3} | {:>12} |{:>15L} |{:>10.0Lf} | {:<}")
{}

void ResultTable::PrintRow(const ResultTimeNodes& rtn, int true_score)
{
	rows.emplace_back(rtn.result.score, rtn.time, rtn.nodes);
	scores.push_back(true_score);

	std::string depth = std::to_string(rtn.result.depth);
	if (rtn.result.confidence_level < std::numeric_limits<float>::infinity())
		depth += "@" + std::format("{:.1f}", rtn.result.confidence_level);
	std::string eval = ScoreToString(rtn.result.score);
	std::string score = (rtn.result.score == true_score) ? "" : ScoreToString(true_score);
	std::string time = RichTimeFormat(rtn.time);
	double nps = static_cast<double>(rtn.nodes) / rtn.time.count();
	std::string pv = to_string(rtn.result.best_move);

	Table::PrintRow(rows.size(), depth, eval, score, time, rtn.nodes, nps, pv);
}

void ResultTable::PrintRow(const ResultTimeNodes& rtn)
{
	PrintRow(rtn, rtn.result.score);
}

void ResultTable::PrintRow(const ScoreTimeNodes& stn, int true_score)
{
	rows.push_back(stn);
	scores.push_back(true_score);

	std::string eval = ScoreToString(stn.score);
	std::string score = (stn.score == true_score) ? "" : ScoreToString(true_score);
	std::string time = RichTimeFormat(stn.time);
	double nps = static_cast<double>(stn.nodes) / stn.time.count();

	Table::PrintRow(rows.size(), "", eval, score, time, stn.nodes, nps, "");
}

void ResultTable::PrintRow(const ScoreTimeNodes& stn)
{
	PrintRow(stn, stn.score);
}

void ResultTable::PrintSummary() const
{
	int64_t abs_err = 0;
	uint64_t nodes = 0;
	std::chrono::duration<double> time;
	for (int i = 0; i < rows.size(); i++)
	{
		abs_err += std::abs(rows[i].score - scores[i]);
		time += rows[i].time;
		nodes += rows[i].nodes;
	}
	double avg_abs_err = static_cast<double>(abs_err) / static_cast<double>(rows.size());
	double nps = static_cast<double>(nodes) / time.count();

	if (avg_abs_err == 0)
		std::cout << std::format("                              {:>12}  {:>15L}  {:>10.0Lf}\n", RichTimeFormat(time), nodes, nps);
	else
		std::cout << std::format("          avg_abs_err: {:4.1f}   {:>12}  {:>15L}  {:>10.0Lf}\n", avg_abs_err, RichTimeFormat(time), nodes, nps);
}

void ResultTable::clear()
{
	rows.clear();
	scores.clear();
}