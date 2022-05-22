#pragma once
#include "Core/Core.h"
#include "Search/Search.h"
#include <functional>
#include <string>
#include <set>
#include <optional>

//struct Request // TODO: Rename to SearchRequest
//{
//	Field move = Field::none;
//	Intensity intensity;
//
//	Request(Field move, Intensity intensity) noexcept : move(move), intensity(intensity) {}
//	Request(Field move, int depth, Confidence selectivity = Confidence::Certain()) noexcept : move(move), intensity(depth, selectivity) {}
//	Request(Intensity intensity) noexcept : intensity(intensity) {}
//	Request(int depth, Confidence selectivity = Confidence::Certain()) noexcept : intensity(depth, selectivity) {}
//
//	static Request ExactScore(const Position& pos) { return Request(pos.EmptyCount()); }
//	static std::set<Request> AllMoves(Position);
//
//	bool operator==(const Request&) const noexcept = default;
//	bool operator!=(const Request&) const noexcept = default;
//	auto operator<=>(const Request&) const noexcept = default;
//
//	bool HasMove() const { return move != Field::none; }
//	bool IsCertain() const { return intensity.IsCertain(); }
//	int Depth() const { return intensity.depth; }
//};
//
//inline std::string to_string(const Request& r) { return (r.HasMove() ? to_string(r.move) + " " : std::string{}) + to_string(r.intensity); }


//struct Result // TODO: Rename to SearchResult
//{
//	int score;
//	uint64 nodes = 0;
//	std::chrono::duration<double> duration{ 0 };
//
//	Result(int score) noexcept : score(score) {}
//	Result(int score, uint64 nodes, std::chrono::duration<double> duration) noexcept : score(score), nodes(nodes), duration(duration) {}
//
//	static Result None() noexcept { return { undefined_score, 0, std::chrono::duration<double>(0) }; }
//
//	bool operator==(const Result&) const noexcept = default;
//	bool operator!=(const Result&) const noexcept = default;
//
//	void clear() { *this = None(); }
//	bool HasValue() const { return score != undefined_score; }
//};

class NoMovePuzzle
{
public:
	class Task
	{
	public:
		Intensity request; // TODO: Make private!
		int result; // TODO: Make private!

		Task(Intensity request, int result = undefined_score) noexcept : request(request), result(result) {}

		bool operator==(const Task&) const noexcept = default;
		bool operator!=(const Task&) const noexcept = default;

		int Score() const noexcept { return result; }
		int Depth() const noexcept { return request.depth; }
		Confidence Certainty() const noexcept { return request.certainty; }
		bool IsDone() const noexcept { return result != undefined_score; }
		bool IsCertain() const noexcept { return request.IsCertain(); }
		bool IsExact() const noexcept { return request.IsExact(); }

		void ResolveWith(int score) { result = score; }
		void ClearResult() { result = undefined_score; }

		// "(+00 d7 87%)" or "(-01 d4)" or "(+02)"
		std::string to_string() const;
	};

	Position pos;
	std::vector<Task> tasks;

	NoMovePuzzle(Position pos) noexcept : pos(pos) {}
	NoMovePuzzle(Position pos, Intensity request) noexcept : pos(pos), tasks({ Task{request} }) {}
	NoMovePuzzle(Position pos, Intensity request, int result) noexcept : pos(pos), tasks({ Task{request, result} }) {}
	NoMovePuzzle(Position pos, std::vector<Task> tasks) noexcept : pos(pos), tasks(std::move(tasks)) {}

	static NoMovePuzzle WithExactScore(Position pos, int score) { return { pos, pos.EmptyCount(), score }; }

	bool operator==(const NoMovePuzzle&) const noexcept = default;
	bool operator!=(const NoMovePuzzle&) const noexcept = default;

	void clear() noexcept { tasks.clear(); }
	std::size_t size() const noexcept { return tasks.size(); }
	bool empty() const noexcept { return tasks.empty(); }

	bool contains(const Intensity& request) const;
	void insert(const Intensity& request);
	void insert(const range<Intensity> auto& requests) { for (const Intensity& r : requests) insert(r); }
	void erase(const Intensity& request);
	void erase_if(std::function<bool(const Task&)> pred);
	void erase_undones();
	void erase_inexacts();

	int EmptyCount() const { return pos.EmptyCount(); }

	bool AllTasksDone() const;
	bool AnyTaskDone() const;
	int ResultOf(const Intensity& request) const noexcept(false);
	void ClearResult(const Intensity& request) noexcept(false);
	void ClearInexacts();
	std::set<Intensity> SolvedIntensities() const;
	std::optional<Intensity> MaxSolvedIntensity() const;

	bool Solve(Algorithm&);

	// "--XO-- e20 (+02) (+00 d7 87%)"
	std::string to_string() const;
};

inline int EmptyCount(const NoMovePuzzle& p) { return p.pos.EmptyCount(); }


class AllMovePuzzle
{
public:
	class Task
	{
	public:
		struct SubTask
		{
			Field move;
			int result;

			SubTask(Field move, int result = undefined_score) : move(move), result(result) {}

			bool operator==(const SubTask&) const noexcept = default;
			bool operator!=(const SubTask&) const noexcept = default;

			int Score() const noexcept { return result; }
			bool IsDone() const noexcept { return result != undefined_score; }

			void ResolveWith(int score) { result = score; }
			void ClearResult() { result = undefined_score; }

			// "A1:+00"
			std::string to_string() const;
		};

		Intensity request;
		std::vector<SubTask> results;

		Task(Intensity request, std::vector<SubTask> results) noexcept : request(request), results(std::move(results)) {}
		Task(Intensity request, const Position&) noexcept;

		bool operator==(const Task&) const noexcept = default;
		bool operator!=(const Task&) const noexcept = default;

		Intensity GetIntensity() const { return request; }
		bool IsDone() const noexcept { return ranges::all_of(results, &SubTask::IsDone); }
		bool IsCertain() const noexcept { return request.IsCertain(); }
		bool IsExact() const noexcept { return request.IsExact(); }
		int Depth() const noexcept { return request.depth; }
		Confidence Certainty() const noexcept { return request.certainty; }

		// "d7 87% A1:+00 B7:+02" or "A1:+00 B7:+02"
		std::string to_string() const;
	};

	Position pos;
	std::vector<Task> tasks;

	AllMovePuzzle(Position pos) noexcept : pos(pos) {}
	AllMovePuzzle(Position pos, Intensity request) noexcept : pos(pos), tasks({ Task{request, pos} }) {}
	AllMovePuzzle(Position pos, std::vector<Task> tasks) noexcept : pos(pos), tasks(std::move(tasks)) {}

	bool operator==(const AllMovePuzzle&) const noexcept = default;
	bool operator!=(const AllMovePuzzle&) const noexcept = default;

	void clear() noexcept { tasks.clear(); }
	std::size_t size() const noexcept { return tasks.size(); }
	bool empty() const noexcept { return tasks.empty(); }

	bool contains(const Intensity& request) const;
	void insert(const Intensity& request);
	void insert(const range<Intensity> auto& requests);
	void erase(const Intensity& request);
	void erase_if(std::function<bool(const Task&)> pred);
	void erase_undones();
	void erase_inexacts();

	int EmptyCount() const { return pos.EmptyCount(); }

	bool AllTasksDone() const;
	bool AnyTaskDone() const;
	std::vector<Task::SubTask> ResultOf(const Intensity& request) const noexcept(false);
	std::set<Intensity> SolvedIntensities() const;
	std::optional<Intensity> MaxSolvedIntensity() const;

	bool Solve(Algorithm&);

	// "--XO-- e20 (+02) (-01 d2) (+00 d7 87%)"
	std::string to_string() const;
};

inline int EmptyCount(const AllMovePuzzle& p) { return p.pos.EmptyCount(); }

//class Puzzle
//{
//public:
//	class Task
//	{
//		Request request;
//		Result result;
//	public:
//		Task(Request request, Result result = Result::None()) noexcept : request(request), result(result) {}
//
//		bool operator==(const Task&) const noexcept = default;
//		bool operator!=(const Task&) const noexcept = default;
//
//		const Request& GetRequest() const { return request; }
//		const Result& GetResult() const { return result; }
//
//		Field Move() const { return request.move; }
//		Intensity GetIntensity() const { return request.intensity; }
//		int Score() const { return result.score; }
//		uint64 Nodes() const { return result.nodes; }
//		std::chrono::duration<double> Duration() const { return result.duration; }
//
//		bool IsDone() const { return result.HasValue(); }
//		bool HasMove() const { return request.HasMove(); }
//		bool IsCertain() const { return request.IsCertain(); }
//		int Depth() const { return request.Depth(); }
//
//		void ResolveWith(const ::Result& r) { result = r; }
//		void ResolveWith(int score, uint64 nodes, std::chrono::duration<double> duration) { ResolveWith({ score, nodes, duration }); }
//		void ResolveWith(int score) { ResolveWith({ score }); }
//		void ClearResult() { result.clear(); }
//	};
//
//	Position pos;
//	std::vector<Task> tasks;
//
//	Puzzle(Position pos) noexcept : pos(pos) {}
//	Puzzle(Position pos, Request request, Result result = Result::None()) noexcept : pos(pos), tasks({ Task{request, result} }) {}
//	Puzzle(Position pos, std::vector<Task> tasks) noexcept : pos(pos), tasks(std::move(tasks)) {}
//
//	static Puzzle WithExactScore(Position pos, int score) { return { pos, pos.EmptyCount(), score }; }
//	static Puzzle WithAllMoves(Position);
//
//	bool operator==(const Puzzle&) const noexcept = default;
//	bool operator!=(const Puzzle&) const noexcept = default;
//
//	void clear() noexcept { tasks.clear(); }
//	std::size_t size() const noexcept { return tasks.size(); }
//	bool empty() const noexcept { return tasks.empty(); }
//
//	void insert(const Request&);
//	void erase(const Request&);
//	void erase_if(std::function<bool(const Task&)> pred);
//	void erase_undones();
//	void erase_inexacts();
//
//	void ClearResult(const Request&);
//	void ClearResultIf(std::function<bool(const Task&)> pred);
//
//	uint64 Nodes() const;
//	std::chrono::duration<double> Duration() const;
//	int EmptyCount() const { return pos.EmptyCount(); }
//
//	bool AllTasksDone() const;
//	bool AnyTaskDone() const;
//	bool Contains(const Request&) const;
//	bool Contains(Field move) const; // TODO: Remove?
//	Result ResultOf(const Request&) const noexcept(false);
//	std::set<Field> BestMoves() const;
//	std::set<Intensity> SolvedIntensities() const; // TODO: Add test!
//	std::set<Intensity> SolvedIntensitiesOfAllMoves() const; // TODO: Add test!
//	std::optional<Intensity> MaxSolvedIntensity() const; // TODO: Add test!
//	std::optional<Intensity> MaxSolvedIntensityOfAllMoves() const; // TODO: Add test!
//	std::optional<int> MaxSolvedIntensityScore() const; // TODO: Add test!
//
//	bool Solve(const Algorithm&);
//
//	std::string to_string() const;
//};

//template <typename T>
//uint64 Nodes(const T& t) requires requires { t.Nodes(); }
//{
//	return t.Nodes();
//}
//
////template <typename T>
////requires requires (const T& t) { Nodes(t); }
//uint64 Nodes(const ranges::range auto& R)
//{
//	return std::transform_reduce(R.begin(), R.end(),
//		uint64(0), std::plus(),
//		[](const auto& t) { return Nodes(t); });
//}
//
//template <typename T>
//std::chrono::duration<double> Duration(const T& t) requires requires { t.Duration(); }
//{
//	return t.Duration();
//}
//
////template <typename T>
////requires requires (const T& t) { Duration(t); }
//std::chrono::duration<double> Duration(const ranges::range auto& R)
//{
//	return std::transform_reduce(R.begin(), R.end(),
//		std::chrono::duration<double>(0), std::plus(),
//		[](const auto& t) { return Duration(t); });
//}

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
