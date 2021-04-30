#include "Puzzle.h"

Puzzle Puzzle::Certain(::Position pos, int min_depth, int max_depth)
{
	Puzzle puzzle(pos);
	for (int d = min_depth; d <= max_depth; d++)
		puzzle.Add(Search::Request::Certain(d));
	return puzzle;
}

int Puzzle::Score() const
{
	if (not IsSolved())
		throw std::runtime_error("Puzzle is not solved.");
	if (result.empty())
		throw std::runtime_error("Puzzle has no result.");
	return result.back().window.lower();
}

void Puzzle::Solve(const Search::Algorithm& algorithm, const std::function<void(const Puzzle&)>& request_completion_task)
{
	auto alg = algorithm.Clone();

	for (const Search::Request& req : request)
	{
		const auto start = std::chrono::high_resolution_clock::now();
		result.push_back(alg->Eval(pos, req));
		const auto stop = std::chrono::high_resolution_clock::now();

		node_count = alg->node_count;
		duration += stop - start;

		request_completion_task(*this);
	}
}

Project::Project(const Project& o) noexcept : Project(o.puzzle)
{
	request_completion_task = o.request_completion_task;
	puzzle_completion_task = o.puzzle_completion_task;
	project_completion_task = o.project_completion_task;
}

Project::Project(Project&& o) noexcept : Project(o.puzzle)
{
	request_completion_task = std::move(o.request_completion_task);
	puzzle_completion_task = std::move(o.puzzle_completion_task);
	project_completion_task = std::move(o.project_completion_task);
}

Project& Project::operator=(const Project& o) noexcept
{
	puzzle = o.puzzle;
	request_completion_task = o.request_completion_task;
	puzzle_completion_task = o.puzzle_completion_task;
	project_completion_task = o.project_completion_task;
	return *this;
}

Project& Project::operator=(Project&& o) noexcept
{
	puzzle = std::move(o.puzzle);
	request_completion_task = std::move(o.request_completion_task);
	puzzle_completion_task = std::move(o.puzzle_completion_task);
	project_completion_task = std::move(o.project_completion_task);
	return *this;
}

uint64 Project::Nodes() const
{
	return std::transform_reduce(puzzle.begin(), puzzle.end(), 
								 0ULL, 
								 std::plus(), 
								 [](const Puzzle& p) { return p.Nodes(); });
}

std::chrono::duration<double> Project::Duration() const
{
	return std::transform_reduce(puzzle.begin(), puzzle.end(), 
								 std::chrono::duration<double>(0), 
								 std::plus(), 
								 [](const Puzzle& p) { return p.Duration(); });
}

bool Project::SolveNext(const Search::Algorithm& alg, bool force)
{
	std::size_t index = next++;
	if (index >= puzzle.size())
		return false;
	if (not force and puzzle[index].IsSolved())
		return false;
	puzzle[index].Solve(alg, request_completion_task);
	solved++;

	puzzle_completion_task(puzzle[index]);
	if (IsSolved())
		project_completion_task(*this);
	return true;
}

uint64 ProjectDB::Nodes() const
{
	return std::transform_reduce(projects.begin(), projects.end(), 
								 0ULL, 
								 std::plus(), 
								 [](const Project& p) { return p.Nodes(); });
}

std::chrono::duration<double> ProjectDB::Duration() const
{
	return std::transform_reduce(projects.begin(), projects.end(), 
								 std::chrono::duration<double>(0), 
								 std::plus(), 
								 [](const Project& p) { return p.Duration(); });
}

bool ProjectDB::IsSolved() const
{
	return std::all_of(projects.begin(), projects.end(),
					   [](const Project& p){ return p.IsSolved(); });
}

void ProjectDB::SolveNext(const Search::Algorithm& alg)
{
	for (auto& p : projects)
		if (p.SolveNext(alg))
			return;
}
