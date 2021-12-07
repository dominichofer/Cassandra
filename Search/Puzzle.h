#pragma once
#include "Core/Core.h"
#include "Executor.h"
#include "Objects.h"
#include "Algorithm.h"
#include <chrono>
#include <execution>
#include <functional>
#include <string>
#include <set>
#include <ranges>
#include <range/v3/view/filter.hpp>

struct Request
{
	Field move = Field::none;
	Intensity intensity;

	Request(Field move, Intensity intensity) noexcept : move(move), intensity(intensity) {}
	Request(Field move, int depth, Confidence selectivity = Confidence::Certain()) noexcept : move(move), intensity(depth, selectivity) {}
	Request(Intensity intensity) noexcept : intensity(intensity) {}
	Request(int depth, Confidence selectivity = Confidence::Certain()) noexcept : intensity(depth, selectivity) {}

	static Request ExactScore(const Position& pos) { return Request(pos.EmptyCount()); }
	static std::set<Request> AllMoves(Position);

	bool operator==(const Request&) const noexcept = default;
	bool operator!=(const Request&) const noexcept = default;
	auto operator<=>(const Request&) const noexcept = default;

	bool HasMove() const { return move != Field::none; }
};

inline std::string to_string(const Request& r) { return (r.HasMove() ? to_string(r.move) + " " : std::string{}) + to_string(r.intensity); }


struct Result
{
	int score = undefined_score;
	uint64 nodes = 0;
	std::chrono::duration<double> duration{ 0 };

	Result() noexcept = default;
	Result(int score) noexcept : score(score) {}
	Result(int score, uint64 nodes, std::chrono::duration<double> duration) noexcept : score(score), nodes(nodes), duration(duration) {}

	bool operator==(const Result&) const noexcept = default;
	bool operator!=(const Result&) const noexcept = default;

	bool HasValue() const { return score != undefined_score; }
};

class Puzzle
{
public:
	class Task
	{
		Request request;
		Result result;
	public:
		Task(Request request, Result result = {}) noexcept : request(request), result(result) {}

		bool operator==(const Task&) const noexcept = default;
		bool operator!=(const Task&) const noexcept = default;

		const Request& GetRequest() const noexcept { return request; }
		const Result& GetResult() const noexcept { return result; }

		 Field Move() const { return request.move; }
		 Intensity GetIntensity() const { return request.intensity; }
		 int Score() const { return result.score; }
		 uint64 Nodes() const { return result.nodes; }
		 std::chrono::duration<double> Duration() const { return result.duration; }

		bool IsDone() const { return result.HasValue(); }
		bool HasMove() const { return request.HasMove(); }

		void ResolveWith(const ::Result& r) { result = r; }
		void ResolveWith(int score, uint64 nodes, std::chrono::duration<double> duration) { ResolveWith({ score, nodes, duration }); }
		void ResolveWith(int score) { ResolveWith({ score }); }
		void ClearResult() { result = ::Result{}; }
	};

	Position pos;
	std::vector<Task> tasks;

	Puzzle(Position pos) noexcept : pos(pos) {}
	Puzzle(Position pos, Request request, Result result = {}) noexcept : pos(pos), tasks({ Task{request, result} }) {}
	Puzzle(Position pos, std::vector<Task> tasks) noexcept : pos(pos), tasks(std::move(tasks)) {}

	static Puzzle WithExactScore(Position pos, int score) { return { pos, pos.EmptyCount(), score }; }
	static Puzzle WithAllMoves(Position);

	bool operator==(const Puzzle&) const noexcept = default;
	bool operator!=(const Puzzle&) const noexcept = default;

	void clear() noexcept { tasks.clear(); }
	std::size_t size() const noexcept { return tasks.size(); }
	bool empty() const noexcept { return tasks.empty(); }

	void insert(const Request&);
	template <std::ranges::range R>
	requires std::is_same_v<std::ranges::range_value_t<R>, Request>
	void insert(const Request&);
	void erase(const Request&);
	void erase_if(std::function<bool(const Task&)> pred);
	void erase_if_undone();

	void ClearResult(const Request&);
	void ClearResultIf(std::function<bool(const Task&)> pred);

	uint64 Nodes() const;
	std::chrono::duration<double> Duration() const;

	bool AllTasksDone() const;
	bool AnyTaskDone() const;
	bool Contains(const Request&) const;
	bool Contains(Field move) const; // TODO: Remove?
	Result ResultOf(const Request&) const noexcept(false);
	std::set<Field> BestMoves() const;
	std::set<Intensity> SolvedIntensities() const; // TODO: Add test!
	std::set<Intensity> SolvedIntensitiesOfAllMoves() const; // TODO: Add test!
	std::optional<Intensity> MaxSolvedIntensity() const; // TODO: Add test!
	std::optional<Intensity> MaxSolvedIntensityOfAllMoves() const; // TODO: Add test!
	std::optional<int> MaxSolvedIntensityScore() const; // TODO: Add test!

	bool Solve(const Search::Algorithm&);

	// "--XO-- (+01) (A1 : +00) (B5 d7 87% : +01)"
	std::string to_string() const;
};

inline std::string to_string(const Puzzle& puzzle) { return puzzle.to_string(); }
inline std::ostream& operator<<(std::ostream& os, const Puzzle& puzzle) { return os << to_string(puzzle); }

inline uint64 Nodes(const Puzzle& p) { return p.Nodes(); }
inline std::chrono::duration<double> Duration(const Puzzle& p) { return p.Duration(); }

inline int EmptyCount(const Puzzle& p) { return p.pos.EmptyCount(); }

template <typename T>
concept PuzzleRange = std::ranges::range<T> and std::is_same_v<std::ranges::range_value_t<T>, Puzzle>; // TODO: Replace by range<Puzzle>?

namespace views
{
	inline auto empty_count_filter(int empty_count) { return ranges::views::filter([empty_count](const Puzzle& p) { return p.pos.EmptyCount() == empty_count; }); }
}

inline uint64 Nodes(const range<Puzzle> auto& puzzles)
{
	return std::transform_reduce(puzzles.begin(), puzzles.end(),
		0ULL, std::plus(),
		[](const Puzzle& p) { return p.Nodes(); });
}

inline std::chrono::duration<double> Duration(const range<Puzzle> auto& puzzles)
{
	return std::transform_reduce(puzzles.begin(), puzzles.end(),
		std::chrono::duration<double>(0), std::plus(),
		[](const Puzzle& p) { return p.Duration(); });
}

//class PuzzleProject final : public TaskLibrary<Puzzle>
//{
//public:
//	PuzzleProject() noexcept = default;
//	template <typename Iterator>
//	PuzzleProject(const Iterator& begin, const Iterator& end) noexcept : TaskLibrary<Puzzle>(begin, end) {}
//	PuzzleProject(std::vector<Puzzle> wu) noexcept : TaskLibrary<Puzzle>(std::move(wu)) {}
//	PuzzleProject(const TaskLibrary<Puzzle>& o) noexcept : TaskLibrary<Puzzle>(o) {}
//
//	uint64 Nodes() const;
//	std::chrono::duration<double> Duration() const;
//
//	void MakeAllHave(const ::Request&);
//	void erase(const ::Request& r) { for (auto& x : wu) ::erase(x, r); }
//	void PrepareForTests();
//
//	// thread-safe
//	//std::size_t Scheduled(const Request&) const { return next.load(std::memory_order_acquire); }
//	//std::size_t Processed(const Request&) const { return processed.load(std::memory_order_acquire); }
//	//bool HasWork() const { return Scheduled() < wu.size(); }
//	//bool IsDone() const { return Processed() == wu.size(); }
//};
//
//
//
//uint64 Nodes(const PuzzleProject&);
//uint64 Nodes(const std::vector<PuzzleProject>&);
//
//std::chrono::duration<double> Duration(const PuzzleProject&);
//std::chrono::duration<double> Duration(const std::vector<PuzzleProject>&);
//
//void MakeAllHave(PuzzleProject&, const ::Request&);
//void MakeAllHave(std::vector<PuzzleProject>&, const ::Request&);
