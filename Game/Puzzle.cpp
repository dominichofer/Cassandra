#include "Puzzle.h"
#include <algorithm>

//std::set<Request> Request::AllMoves(Position pos)
//{
//	std::set<Request> set;
//	for (Field move : PossibleMoves(pos))
//		set.insert({ move, pos.EmptyCount() - 1 });
//	return set;
//}

//Puzzle Puzzle::WithAllMoves(Position pos)
//{
//	Puzzle ret(pos);
//	for (const Request& r : Request::AllMoves(pos))
//		ret.insert(r);
//	return ret;
//}
//
//void Puzzle::insert(const Request& r)
//{
//	if (not Contains(r))
//		tasks.push_back(r);
//}
//
//void Puzzle::erase(const Request& suspect)
//{
//	std::erase_if(tasks, [&suspect](const Task& t) { return t.GetRequest() == suspect; });
//}
//
//void Puzzle::erase_if(std::function<bool(const Task&)> pred)
//{
//	std::erase_if(tasks, pred);
//}
//
//void Puzzle::erase_undones()
//{
//	std::erase_if(tasks, [](const Task& task) { return not task.IsDone(); });
//}
//
//void Puzzle::erase_inexacts()
//{
//	std::erase_if(tasks, [this, exact = Request::ExactScore(pos)](const Puzzle::Task& t) { return t.GetRequest() != exact; });
//}
//
//void Puzzle::ClearResult(const Request& r)
//{
//	ClearResultIf([&r](const Task& task) { return task.GetRequest() == r; });
//}
//
//void Puzzle::ClearResultIf(std::function<bool(const Task&)> pred)
//{
//	for (auto& t : tasks)
//		if (pred(t))
//			t.ClearResult();
//}
//
//uint64 Puzzle::Nodes() const
//{
//	return std::transform_reduce(tasks.begin(), tasks.end(),
//		0ULL, std::plus(),
//		[](const Task& t) { return t.Nodes(); });
//}
//
//std::chrono::duration<double> Puzzle::Duration() const
//{
//	return std::transform_reduce(tasks.begin(), tasks.end(),
//		std::chrono::duration<double>{ 0 }, std::plus(),
//		[](const Task& t) { return t.Duration(); });
//}
//
//bool Puzzle::AllTasksDone() const
//{
//	return std::ranges::all_of(tasks,
//		[](const Task& t) { return t.IsDone(); });
//}
//
//bool Puzzle::AnyTaskDone() const
//{
//	return std::ranges::any_of(tasks,
//		[](const Task& t) { return t.IsDone(); });
//}
//
//bool Puzzle::Contains(const Request& suspect) const
//{
//	return std::ranges::any_of(tasks,
//		[&suspect](const Task& t) { return t.GetRequest() == suspect; });
//}
//
//bool Puzzle::Contains(Field move) const
//{
//	return std::ranges::any_of(tasks,
//		[move](const Task& task) { return task.Move() == move; });
//}
//
//Result Puzzle::ResultOf(const Request& suspect) const noexcept(false)
//{
//	auto it = std::ranges::find_if(tasks,
//		[&suspect](const Task& task) { return task.GetRequest() == suspect; });
//	if (it != tasks.end())
//		return it->GetResult();
//	throw std::runtime_error("Puzzle does not contain request.");
//}
//
//std::set<Field> Puzzle::BestMoves() const
//{
//	std::optional<Intensity> opt_max = MaxSolvedIntensityOfAllMoves();
//	if (not opt_max.has_value())
//		return {};
//	Intensity max_intensity = opt_max.value();
//
//	std::set<Field> best_moves;
//	int min_score = inf_score;
//	for (Field move : PossibleMoves(pos))
//	{
//		Result r = ResultOf(Request{ move, max_intensity - 1 });
//		if (r.score < min_score)
//		{
//			best_moves.clear();
//			min_score = r.score;
//		}
//		if (r.score == min_score)
//			best_moves.insert(move);
//	}
//	return best_moves;
//}
//
//std::set<Intensity> Puzzle::SolvedIntensities() const
//{
//	std::set<Intensity> set = SolvedIntensitiesOfAllMoves();
//
//	// Add all solved intensities from tasks with no moves
//	for (const Task& t : tasks)
//		if (t.IsDone() and not t.HasMove())
//			set.insert(t.GetIntensity());
//
//	return set;
//}
//
//std::set<Intensity> Puzzle::SolvedIntensitiesOfAllMoves() const
//{
//	std::set<Intensity> set;
//
//	// Add all solved intensities from tasks with moves
//	for (const Task& t : tasks)
//		if (t.IsDone() and t.HasMove())
//			set.insert(t.GetIntensity() + 1);
//
//	// Filter out intensities that aren't solved in all possible moves
//	for (Field move : PossibleMoves(pos))
//		std::erase_if(set,
//			[&](const Intensity& i) {
//				Request r{ move, i - 1 };
//				return not Contains(r) or not ResultOf(r).HasValue();
//			});
//
//	return set;
//}
//
//std::optional<Intensity> Puzzle::MaxSolvedIntensity() const
//{
//	std::set<Intensity> solved = SolvedIntensities();
//	if (solved.empty())
//		return std::nullopt;
//	return std::ranges::max(solved);
//}
//
//std::optional<Intensity> Puzzle::MaxSolvedIntensityOfAllMoves() const
//{
//	std::set<Intensity> solved_intensities = SolvedIntensitiesOfAllMoves();
//	if (solved_intensities.empty())
//		return std::nullopt;
//	return std::ranges::max(solved_intensities);
//}
//
//// Used in pattern fitting
//std::optional<int> Puzzle::MaxSolvedIntensityScore() const
//{
//	std::optional<Intensity> opt_max = MaxSolvedIntensity();
//	if (not opt_max.has_value())
//		return std::nullopt;
//	Intensity max = opt_max.value();
//
//	if (Contains(max))
//		return ResultOf(max).score;
//
//	Moves pm = PossibleMoves(pos);
//	return ranges::max(
//		tasks
//		| ranges::views::filter([&](const Task& t) { return t.IsDone() and t.HasMove() and pm.contains(t.Move()) and t.GetIntensity() == max - 1; })
//		| ranges::views::transform([&](const Task& t) { return -t.Score(); }));
//
//
//	//int max_score = -inf_score;
//	//Intensity max_intensity{ -1 };
//
//	//std::optional<IntensityResult> ret = std::nullopt;
//	//auto cmp = [](const Task& l, const Task& r) { return l.GetIntensity() < r.GetIntensity(); };
//	//auto possible_moves = PossibleMoves(pos);
//	//if (possible_moves and ContainsAllPossibleMovesDone())
//	//{
//	//	IntensityResult ir{ Intensity::Exact(pos), Result(-inf_score) };
//	//	for (Field move : possible_moves)
//	//	{
//	//		Task t = std::ranges::max(
//	//			tasks | std::views::filter([move](const Task& t) { return t.Move() == move and t.IsDone(); }),
//	//			cmp);
//	//		ir.intensity = std::min(ir.intensity, t.GetIntensity());
//	//		ir.result.score = std::max(ir.result.score, -t.Score());
//	//		ir.result.nodes += t.Nodes();
//	//		ir.result.duration += t.Duration();
//	//	}
//	//	ret = ir;
//	//}
//
//	//auto done_tasks_with_no_moves = tasks | std::views::filter([](const Task& t) { return t.IsDone() and not t.HasMove(); });
//	//if (done_tasks_with_no_moves)
//	//{
//	//	Task t = std::ranges::max(done_tasks_with_no_moves, cmp);
//	//	if (not ret.has_value() or t.GetIntensity() >= ret.value().intensity)
//	//		ret = IntensityResult(t.GetIntensity(), t.Result());
//	//}
//	//return ret;
//}
//
//bool Puzzle::Solve(const Algorithm& algorithm)
//{
//	bool had_work = false;
//	for (Task& task : tasks)
//	{
//		if (task.IsDone())
//			continue;
//
//		auto alg = algorithm.Clone();
//		auto position = pos;
//		if (task.HasMove())
//			position = Play(position, task.Move());
//
//		const auto start = std::chrono::high_resolution_clock::now();
//		int score = alg->Eval(position, task.GetIntensity());
//		const auto stop = std::chrono::high_resolution_clock::now();
//
//		task.ResolveWith(score, alg->nodes, stop - start);
//		had_work = true;
//	}
//	return had_work;
//}
//
//std::string Puzzle::to_string() const
//{
//	// Position
//	std::string ret = ::to_string(pos) + " e" + std::to_string(pos.EmptyCount());
//
//	for (const Task& task : tasks)
//	{
//		bool has_move = task.HasMove();
//		auto intensity = task.GetIntensity();
//
//		ret += " (";
//		// move
//		if (has_move)
//			ret += ::to_string(task.Move()) + " ";
//		// score
//		ret += DoubleDigitSignedInt(has_move ? -task.Score() : task.Score());
//		// intensity
//		if ((not intensity.IsCertain()) or intensity.depth < pos.EmptyCount() - (has_move ? 1 : 0))
//			ret += " d" + ::to_string(intensity);
//		ret += ")";
//	}
//	return ret;
//}

bool NoMovePuzzle::contains(const Intensity& suspect) const
{
	return std::ranges::any_of(tasks,
		[&suspect](const Task& t) { return t.request == suspect; });
}

void NoMovePuzzle::insert(const Intensity& request)
{
	if (not contains(request))
		tasks.push_back(request);
}

void NoMovePuzzle::erase(const Intensity& suspect)
{
	std::erase_if(tasks, [&suspect](const Task& t) { return t.request == suspect; });
}

void NoMovePuzzle::erase_if(std::function<bool(const Task&)> pred)
{
	std::erase_if(tasks, pred);
}

void NoMovePuzzle::erase_undones()
{
	std::erase_if(tasks, [](const Task& task) { return not task.IsDone(); });
}

void NoMovePuzzle::erase_inexacts()
{
	std::erase_if(tasks, [this](const Task& t) { return not t.IsExact(); });
}

bool NoMovePuzzle::AllTasksDone() const
{
	return ranges::all_of(tasks, &Task::IsDone);
}

bool NoMovePuzzle::AnyTaskDone() const
{
	return ranges::any_of(tasks, &Task::IsDone);
}

int NoMovePuzzle::ResultOf(const Intensity& suspect) const noexcept(false)
{
	auto it = std::ranges::find_if(tasks,
		[&suspect](const Task& task) { return task.request == suspect; });
	if (it != tasks.end())
		return it->result;
	throw std::runtime_error("Does not contain request.");
}

void NoMovePuzzle::ClearResult(const Intensity& suspect) noexcept(false)
{
	auto it = std::ranges::find_if(tasks,
		[&suspect](const Task& task) { return task.request == suspect; });
	if (it == tasks.end())
		throw std::runtime_error("Does not contain request.");
	it->ClearResult();
}


void NoMovePuzzle::ClearInexacts()
{
	for (Task& t : tasks)
		if (not t.IsExact())
			t.ClearResult();
}

std::set<Intensity> NoMovePuzzle::SolvedIntensities() const
{
	std::set<Intensity> set;
	for (const Task& t : tasks)
		if (t.IsDone())
			set.insert(t.request);
	return set;
}

std::optional<Intensity> NoMovePuzzle::MaxSolvedIntensity() const
{
	std::set<Intensity> solved = SolvedIntensities();
	if (solved.empty())
		return std::nullopt;
	return std::ranges::max(solved);
}

bool NoMovePuzzle::Solve(Algorithm& alg)
{
	bool had_work = false;
	for (Task& task : tasks)
	{
		if (task.IsDone())
			continue;

		task.ResolveWith(alg.Eval(pos, task.request));
		had_work = true;
	}
	return had_work;
}

std::string NoMovePuzzle::Task::to_string() const
{
	std::string ret;
	if (IsDone())
		ret = fmt::format("{:+02}", result);
	else
		ret = " ??";
	if (IsExact())
		return ret;
	else
		return ret + fmt::format(" d{}", ::to_string(request));
}

std::string NoMovePuzzle::to_string() const
{
	return fmt::format("{} e{} ({})", pos, pos.EmptyCount(), fmt::join(tasks, "), ("));
}


AllMovePuzzle::Task::Task(Intensity request, const Position& pos) noexcept : request(request)
{
	for (Field move : PossibleMoves(pos))
		results.emplace_back(move);
}

bool AllMovePuzzle::contains(const Intensity& suspect) const
{
	return std::ranges::any_of(tasks,
		[&suspect](const Task& t) { return t.request == suspect; });
}

void AllMovePuzzle::insert(const Intensity& request)
{
	if (not contains(request))
		tasks.emplace_back(request, pos);
}

void AllMovePuzzle::insert(const range<Intensity> auto& requests)
{
	for (const Intensity& r : requests)
		insert(r);
}

void AllMovePuzzle::erase(const Intensity& suspect)
{
	std::erase_if(tasks, [&suspect](const Task& t) { return t.request == suspect; });
}

void AllMovePuzzle::erase_if(std::function<bool(const Task&)> pred)
{
	std::erase_if(tasks, pred);
}

void AllMovePuzzle::erase_undones()
{
	std::erase_if(tasks, [](const Task& task) { return not task.IsDone(); });
}

void AllMovePuzzle::erase_inexacts()
{
	std::erase_if(tasks, [this](const Task& t) { return not t.IsExact(); });
}

bool AllMovePuzzle::AllTasksDone() const
{
	return ranges::all_of(tasks, &Task::IsDone);
}

bool AllMovePuzzle::AnyTaskDone() const
{
	return ranges::any_of(tasks, &Task::IsDone);
}

std::vector<AllMovePuzzle::Task::SubTask> AllMovePuzzle::ResultOf(const Intensity& suspect) const noexcept(false)
{
	auto it = std::ranges::find_if(tasks,
		[&suspect](const Task& task) { return task.request == suspect; });
	if (it != tasks.end())
		return it->results;
	throw std::runtime_error("Does not contain request.");
}

std::set<Intensity> AllMovePuzzle::SolvedIntensities() const
{
	std::set<Intensity> set;
	for (const Task& t : tasks)
		if (t.IsDone())
			set.insert(t.request);
	return set;
}

std::optional<Intensity> AllMovePuzzle::MaxSolvedIntensity() const
{
	std::set<Intensity> solved = SolvedIntensities();
	if (solved.empty())
		return std::nullopt;
	return std::ranges::max(solved);
}

bool AllMovePuzzle::Solve(Algorithm& alg)
{
	bool had_work = false;
	for (Task& task : tasks)
		for (Task::SubTask& sub : task.results)
		{
			if (sub.IsDone())
				continue;

			sub.ResolveWith(-alg.Eval(Play(pos, sub.move), task.request - 1));
			had_work = true;
		}
	return had_work;
}

std::string AllMovePuzzle::Task::SubTask::to_string() const
{
	std::string ret = ::to_string(move);
	if (IsDone())
		ret += fmt::format(":{:+02}", result);
	else
		ret += ": ??";
	return ret;
}

std::string AllMovePuzzle::Task::to_string() const
{
	if (IsExact())
		return fmt::format("d{} {}", request, fmt::join(results, " "));
	else
		return fmt::format("{}", fmt::join(results, " "));
}

std::string AllMovePuzzle::to_string() const
{
	return fmt::format("{} e{} ({})", pos, pos.EmptyCount(), fmt::join(tasks, "), ("));
}
