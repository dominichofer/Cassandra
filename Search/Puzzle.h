#pragma once
#include "Core/Core.h"
#include "Search.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <execution>
#include <functional>
#include <thread>

class Puzzle
{
private:
	Position pos;
	uint64 node_count = 0;
	std::chrono::duration<double> duration{0};
	std::vector<Search::Request> request;
	std::vector<Search::Result> result;
public:
	Puzzle(Position pos) noexcept : pos(pos) {}
	Puzzle(Position pos, Search::Request r) noexcept : pos(pos), request({r}) {}
	Puzzle(Position pos, std::vector<Search::Request> r) noexcept : pos(pos), request(std::move(r)) {}
	Puzzle(Position pos, Search::Request req, Search::Result res) noexcept : pos(pos), request({req}), result({res}) {}
	static Puzzle Exact(Position pos) { return Puzzle(pos, Search::Request::Exact(pos)); }
	static Puzzle Exact(Position pos, Search::Result r) { return Puzzle(pos, Search::Request::Exact(pos), r); }
	static Puzzle Certain(Position, int min_depth, int max_depth);
	static Puzzle CertainAllDepth(Position pos) { return Certain(pos, 0, pos.EmptyCount()); }

	void Add(Search::Request r) { request.push_back(r); }

	[[nodiscard]] Position Position() const { return pos; }
	[[nodiscard]] Search::Request Request(int i = 0) const { return request[i]; }
	[[nodiscard]] Search::Result Result(int i = 0) const { return result[i]; }
	[[nodiscard]] uint64 Nodes() const { return node_count; }
	[[nodiscard]] std::chrono::duration<double> Duration() const { return duration; }

	[[nodiscard]] bool IsSolved() const { return request.size() == result.size(); }
	void Solve(const Search::Algorithm&);
};

class Project
{
	std::atomic<std::size_t> next = 0;
	std::atomic<std::size_t> solved = 0;
	std::vector<Puzzle> puzzle;
	std::function<void(const std::vector<Puzzle>&)> completion_task;
public:
	using value_type = Puzzle;

	Project(std::vector<Puzzle> puzzle, std::function<void(const std::vector<Puzzle>&)> completion_task) noexcept 
		: puzzle(std::move(puzzle)), completion_task(std::move(completion_task))
	{
		std::size_t count = std::count_if(puzzle.begin(), puzzle.end(), [](const Puzzle& p) { return p.IsSolved(); });
		solved = count;
		next = count;
	}
	Project(std::vector<Puzzle> puzzle) noexcept : Project(std::move(puzzle), [](const std::vector<Puzzle>&){}) {}
	Project(std::function<void(const std::vector<Puzzle>&)> completion_task) noexcept : Project({}, std::move(completion_task)) {}

	Project(Project&& o) noexcept : Project(std::move(o.puzzle), std::move(o.completion_task)) {}

	void push_back(Puzzle p) { puzzle.push_back(std::move(p)); }

	const std::vector<Puzzle>& Puzzles() const { return puzzle; }

	[[nodiscard]] bool IsSolved() const { return solved.load(std::memory_order_acquire) == puzzle.size(); }
	bool SolveNext(const Search::Algorithm&);
	void Solve(const Search::Algorithm&);
};

class ProjectDB
{
	std::vector<Project> project;
public:
	ProjectDB(std::vector<Project> project = {}) noexcept : project(std::move(project)) {}

	void push_back(Project p) { project.push_back(std::move(p)); }

	const std::vector<Project>& Projects() const { return project; }

	[[nodiscard]] bool IsSolved() const;
	void SolveNext(const Search::Algorithm&);

	template <typename ExecutionPolicy>
	void Solve(ExecutionPolicy policy, const Search::Algorithm& alg)
	{
		uint threads = std::is_same_v<ExecutionPolicy, std::execution::parallel_policy> ? std::thread::hardware_concurrency() : 1;
		#pragma omp parallel num_threads(threads)
		{
			for (auto& p : project)
				p.Solve(alg);
		}
	}
};