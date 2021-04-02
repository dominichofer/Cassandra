#include "Puzzle.h"

Puzzle Puzzle::Certain(::Position pos, int min_depth, int max_depth)
{
	Puzzle puzzle(pos);
	for (int d = min_depth; d <= max_depth; d++)
		puzzle.Add(Search::Request::Certain(d));
	return puzzle;
}

void Puzzle::Solve(const Search::Algorithm& algorithm)
{
	auto alg = algorithm.Clone();
	const auto start = std::chrono::high_resolution_clock::now();
	for (const Search::Request& req : request)
		result.push_back(alg->Eval(pos, req));
	const auto stop = std::chrono::high_resolution_clock::now();

	node_count = alg->node_count;
	duration = stop - start;
}

bool Project::SolveNext(const Search::Algorithm& alg)
{
	std::size_t index = next++;
	if (index >= puzzle.size())
		return false;
	puzzle[index].Solve(alg);
	solved++;
	if (IsSolved())
		completion_task(puzzle);
	return true;
}

void Project::Solve(const Search::Algorithm& alg)
{
	while (SolveNext(alg))
		continue;
}

bool ProjectDB::IsSolved() const
{
	return std::all_of(project.begin(), project.end(), [](const Project& p){ return p.IsSolved(); });
}

void ProjectDB::SolveNext(const Search::Algorithm& alg)
{
	for (auto& p : project)
		if (p.SolveNext(alg))
			return;
}
