#pragma once
#include "Core/Core.h"
#include "Search.h"
#include <atomic>
#include <chrono>
#include <execution>
#include <functional>

class Puzzle
{
	Position pos;
	uint64 node_count = 0;
	std::chrono::duration<double> duration{0};
	std::vector<Search::Request> request;
	std::vector<Search::Result> result;
public:
	Puzzle(Position pos) noexcept : pos(pos) {}
	Puzzle(Position pos, Search::Request request) noexcept : pos(pos), request({request}) {}
	Puzzle(Position pos, Search::Request request, Search::Result result) noexcept : pos(pos), request({request}), result({result}) {}
	Puzzle(Position pos, std::vector<Search::Request> request) noexcept : pos(pos), request(std::move(request)) {}
	Puzzle(Position pos, std::vector<Search::Request> request, std::vector<Search::Result> result) noexcept
		: pos(pos), request(std::move(request)), result(std::move(result)) {}
	Puzzle(Position pos, uint64 node_count, std::chrono::duration<double> duration, std::vector<Search::Request> request, std::vector<Search::Result> result) noexcept
		: pos(pos), node_count(node_count), duration(duration), request(std::move(request)), result(std::move(result)) {}

	static Puzzle Exact(Position pos) { return Puzzle(pos, Search::Request::Exact(pos)); }
	static Puzzle Exact(Position pos, Search::Result r) { return Puzzle(pos, Search::Request::Exact(pos), r); }
	static Puzzle Exact(Position pos, int score) { return Puzzle(pos, Search::Request::Exact(pos), Search::Result::Exact(pos, score)); }
	static Puzzle Certain(Position, int min_depth, int max_depth);
	static Puzzle CertainAllDepths(Position pos) { return Certain(pos, 0, pos.EmptyCount()); }

	void Add(const Search::Request& r) { request.push_back(r); }
	void Add(const Search::Result& r) { result.push_back(r); }
	void Add(const Search::Request& req, const Search::Result& res) { Add(req); Add(res); }

	[[nodiscard]] const Position& Position() const { return pos; }
	[[nodiscard]] const Search::Request& Request(int i = 0) const { return request[i]; }
	[[nodiscard]] const Search::Result& Result(int i = 0) const { return result[i]; }
	[[nodiscard]] uint64 Nodes() const { return node_count; }
	[[nodiscard]] const std::chrono::duration<double>& Duration() const { return duration; }

	[[nodiscard]] int Score() const;

	[[nodiscard]] bool IsSolved() const { return result.size() >= request.size(); }
	void Solve(const Search::Algorithm&, const std::function<void(const Puzzle&)>& request_completion_task = [](const Puzzle&){});
};

class Project
{
	std::atomic<std::size_t> next = 0;
	std::atomic<std::size_t> solved = 0;
	std::function<void(const Puzzle&)> request_completion_task = [](const Puzzle&){};
	std::function<void(const Puzzle&)> puzzle_completion_task = [](const Puzzle&){};
	std::function<void(const Project&)> project_completion_task = [](const Project&){};
	std::vector<Puzzle> puzzle;
public:
	Project() noexcept = default;
	template <typename Iterator>
	Project(const Iterator& begin, const Iterator& end) noexcept : puzzle(begin, end) {}
	Project(std::vector<Puzzle> puzzle) noexcept : puzzle(std::move(puzzle)) {}
	Project(const Project&) noexcept;
	Project(Project&&) noexcept;
	Project& operator=(const Project&) noexcept;
	Project& operator=(Project&&) noexcept;
	~Project() = default;

	using value_type = Puzzle;
	void push_back(const Puzzle& p) { puzzle.push_back(p); }
	void push_back(Puzzle&& p) { puzzle.push_back(std::move(p)); }
	[[nodiscard]] std::size_t size() const noexcept { return puzzle.size(); }

	[[nodiscard]] uint64 Nodes() const;
	[[nodiscard]] std::chrono::duration<double> Duration() const;

	void SetRequestCompletionTask(std::function<void(const Puzzle&)> task) { request_completion_task = std::move(task); }
	void SetPuzzleCompletionTask(std::function<void(const Puzzle&)> task) { puzzle_completion_task = std::move(task); }
	void SetProjectCompletionTask(std::function<void(const Project&)> task) { project_completion_task = std::move(task); }

	const std::vector<Puzzle>& Puzzles() const { return puzzle; }

	[[nodiscard]] bool IsSolved() const { return solved.load(std::memory_order_acquire) == puzzle.size(); }
	bool SolveNext(const Search::Algorithm&, bool force = false);

	template <typename ExecutionPolicy>
	void SolveAll(ExecutionPolicy policy, const Search::Algorithm& alg, bool force = false)
	{
		unsigned int threads = std::is_same_v<ExecutionPolicy, std::execution::parallel_policy> ? std::thread::hardware_concurrency() : 1;
		#pragma omp parallel num_threads(threads)
		{
			while (SolveNext(alg, force))
				continue;
		}
	}
	void SolveAll(const Search::Algorithm& alg, bool force = false) { SolveAll(std::execution::seq, alg, force); }
};

class ProjectDB
{
	std::vector<Project> projects;
public:
	ProjectDB() noexcept = default;
	ProjectDB(std::vector<Project>&& projects) noexcept : projects(std::move(projects)) {}

	void push_back(Project&& p) { projects.push_back(std::move(p)); }
	[[nodiscard]] std::size_t size() const noexcept { return projects.size(); }

	[[nodiscard]] uint64 Nodes() const;
	[[nodiscard]] std::chrono::duration<double> Duration() const;

	void SetRequestCompletionTask(std::function<void(const Puzzle&)> task) { for (auto& p : projects) p.SetRequestCompletionTask(task); }
	void SetPuzzleCompletionTask(std::function<void(const Puzzle&)> task) { for (auto& p : projects) p.SetPuzzleCompletionTask(task); }
	void SetProjectCompletionTask(std::function<void(const Project&)> task) { for (auto& p : projects) p.SetProjectCompletionTask(task); }

	const std::vector<Project>& Projects() const { return projects; }

	[[nodiscard]] bool IsSolved() const;
	void SolveNext(const Search::Algorithm&);

	template <typename ExecutionPolicy>
	void SolveAll(ExecutionPolicy policy, const Search::Algorithm& alg, bool force = false)
	{
		unsigned int threads = std::is_same_v<ExecutionPolicy, std::execution::parallel_policy> ? std::thread::hardware_concurrency() : 1;
		#pragma omp parallel num_threads(threads)
		{
			for (auto& p : projects)
				p.SolveAll(alg, force);
		}
	}
	void SolveAll(const Search::Algorithm& alg, bool force = false) { SolveAll(std::execution::seq, alg, force); }
};