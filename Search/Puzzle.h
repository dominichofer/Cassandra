#pragma once
#include "Core/Core.h"
#include "Project.h"
#include "Objects.h"
#include "Algorithm.h"
#include <array>
#include <atomic>
#include <chrono>
#include <execution>
#include <functional>

struct Request
{
	Field move = Field::none;
	Search::Intensity intensity;

	Request(Field move, Search::Intensity intensity) noexcept : move(move), intensity(intensity) {}
	Request(Field move, int depth, Confidence selectivity = Confidence::Certain()) noexcept : move(move), intensity(depth, selectivity) {}
	Request(int depth, Confidence selectivity = Confidence::Certain()) noexcept : intensity(depth, selectivity) {}

	static Request ExactScore(Position pos) { return Request(pos.EmptyCount()); }
	static std::vector<Request> AllDepths(Position);
	static std::vector<Request> AllMoves(Position);

	[[nodiscard]] bool operator==(const Request&) const noexcept = default;
	[[nodiscard]] bool operator!=(const Request&) const noexcept = default;

	[[nodiscard]] bool HasMove() const { return move != Field::none; }
	//[[nodiscard]] Search::Intensity Intensity() const { return { depth, selectivity }; }

	// TODO: Move this to Search?
	//operator Search::Request() const { return Search::Request(intensity); }
};

struct Result
{
	int score = undefined_score;
	uint64 nodes = 0;
	std::chrono::duration<double> duration{ 0 };

	Result() noexcept = default;
	Result(int score) noexcept : score(score) {}
	Result(int score, uint64 nodes, std::chrono::duration<double> duration) noexcept : score(score), nodes(nodes), duration(duration) {}

	[[nodiscard]] bool operator==(const Result&) const noexcept = default;
	[[nodiscard]] bool operator!=(const Result&) const noexcept = default;

	[[nodiscard]] bool HasValue() const { return score != undefined_score; }
};

class Puzzle
{
public:
	struct Task
	{
		Request request;
		Result result;

		Task(Request request, Result result = {}) noexcept : request(request), result(result) {}

		[[nodiscard]] bool operator==(const Task&) const noexcept = default;
		[[nodiscard]] bool operator!=(const Task&) const noexcept = default;

		[[nodiscard]] bool IsDone() const { return result.HasValue(); }
		[[nodiscard]] bool HasMove() const { return request.HasMove(); }
		[[nodiscard]] Field Move() const { return request.move; }
		[[nodiscard]] Search::Intensity Intensity() const { return request.intensity; }
		[[nodiscard]] int Score() const { return result.score; }
		[[nodiscard]] uint64 Nodes() const { return result.nodes; }
		[[nodiscard]] std::chrono::duration<double> Duration() const { return result.duration; }

		void ResolveWith(const Result& r) { result = r; }
		void RemoveResult() { result = Result{}; }
	};

	Position pos;
	std::vector<Task> tasks;

	Puzzle(Position pos) noexcept : pos(pos) {}
	Puzzle(Position pos, std::vector<Task> tasks) noexcept : pos(pos), tasks(std::move(tasks)) {}

	static Puzzle WithExactScore(Position pos, int score) { return Puzzle(pos, { Task(Request(pos.EmptyCount()), score) }); }
	static Puzzle WithExactScoreForTesting(Position pos, int score) { return Puzzle(pos, { Task(Request(pos.EmptyCount()), score), Task(Request(pos.EmptyCount())) }); }
	static Puzzle WithAllDepths(Position);
	static Puzzle WithAllMoves(Position);

	[[nodiscard]] bool operator==(const Puzzle&) const noexcept = default;
	[[nodiscard]] bool operator!=(const Puzzle&) const noexcept = default;

	void push_back(const Request& r) { tasks.emplace_back(r); }
	void push_back(const std::vector<Request>&);
	void clear() { tasks.clear(); }
	[[nodiscard]] std::size_t size() const noexcept { tasks.size(); }
	[[nodiscard]] bool empty() const noexcept { return tasks.empty(); }

	// inserts element if it is not contained yet.
	void insert(const Request& r) { if (not Contains(r)) tasks.emplace_back(r); }
	void erase(const Request&);

	[[nodiscard]] uint64 Nodes() const;
	[[nodiscard]] std::chrono::duration<double> Duration() const;
	[[nodiscard]] bool Contains(const Request&) const;
	[[nodiscard]] Result ResultOf(const Request&) const noexcept(false); // TODO: Add std::optional
	[[nodiscard]] Result ResultOfSecond(const Request&) const noexcept(false);

	[[nodiscard]] bool HasTaskWithoutMove() const;

	// Returns the task with the biggest 'Search::Intensity' that has no move.
	[[nodiscard]] Task MaxIntensity() const noexcept(false);
	[[nodiscard]] Task MaxIntensity(const std::function<bool(const Search::Intensity&, const Search::Intensity&)>& less) const noexcept(false);

	void RemoveAllUndone();
	void RemoveResult(const Request&);

	void DuplicateRequests();
	bool Solve(const Search::Algorithm&);

	// "--XO-- (+01) (A1 : +00) (B5 d7 87% : +01)"
	[[nodiscard]] std::string to_string() const;
};

inline std::size_t erase(Puzzle& puzzle, const Request& r) { puzzle.erase(r); }

inline [[nodiscard]] std::string to_string(const Puzzle& puzzle) { return puzzle.to_string(); }
inline std::ostream& operator<<(std::ostream& os, const Puzzle& puzzle) { return os << to_string(puzzle); }

inline uint64 Nodes(const Puzzle& p) { return p.Nodes(); }
inline std::chrono::duration<double> Duration(const Puzzle& p) { return p.Duration(); }

uint64 Nodes(const std::vector<Puzzle>&);
std::chrono::duration<double> Duration(const std::vector<Puzzle>&);


//class PuzzleProject final : public Project<Puzzle>
//{
//public:
//	PuzzleProject() noexcept = default;
//	template <typename Iterator>
//	PuzzleProject(const Iterator& begin, const Iterator& end) noexcept : Project<Puzzle>(begin, end) {}
//	PuzzleProject(std::vector<Puzzle> wu) noexcept : Project<Puzzle>(std::move(wu)) {}
//	PuzzleProject(const Project<Puzzle>& o) noexcept : Project<Puzzle>(o) {}
//
//	[[nodiscard]] uint64 Nodes() const;
//	[[nodiscard]] std::chrono::duration<double> Duration() const;
//
//	void MakeAllHave(const ::Request&);
//	void erase(const ::Request& r) { for (auto& x : wu) ::erase(x, r); }
//	void PrepareForTests();
//
//	// thread-safe
//	//[[nodiscard]] std::size_t Scheduled(const Request&) const { return next.load(std::memory_order_acquire); }
//	//[[nodiscard]] std::size_t Processed(const Request&) const { return processed.load(std::memory_order_acquire); }
//	//[[nodiscard]] bool HasWork() const { return Scheduled() < wu.size(); }
//	//[[nodiscard]] bool IsDone() const { return Processed() == wu.size(); }
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
