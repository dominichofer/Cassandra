#pragma once
#include "Search/Search.h"
#include "Table.h"
#include <vector>

class ResultTable : public Table
{
	std::vector<ScoreTimeNodes> rows;
	std::vector<int> scores;
public:
	ResultTable();

	void PrintRow(const ResultTimeNodes&, int score);
	void PrintRow(const ResultTimeNodes&);
	void PrintRow(const ScoreTimeNodes&, int score);
	void PrintRow(const ScoreTimeNodes&);
	void PrintSummary() const;
	void clear();
};