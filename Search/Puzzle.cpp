#include "Puzzle.h"

std::vector<Request> Request::AllDepths(Position pos)
{
	const auto empty_count = pos.EmptyCount();
	std::vector<Request> requests;
	requests.reserve(empty_count + 1);
	for (int d = 0; d <= empty_count; d++)
		requests.emplace_back(d);
	return requests;
}

std::vector<Request> Request::AllMoves(Position pos)
{
	const auto possible_moves = PossibleMoves(pos);
	std::vector<Request> requests;
	requests.reserve(possible_moves.size());
	for (Field move : possible_moves)
		requests.emplace_back(move, pos.EmptyCount() - 1);
	return requests;
}

Puzzle Puzzle::WithAllDepths(Position pos)
{
	Puzzle puzzle(pos);
	puzzle.push_back(Request::AllDepths(pos));
	return puzzle;
}

Puzzle Puzzle::WithAllMoves(Position pos)
{
	Puzzle puzzle(pos);
	puzzle.push_back(Request::AllMoves(pos));
	return puzzle;
}

void Puzzle::push_back(const std::vector<Request>& requests)
{
	for (const Request& r : requests)
		push_back(r);
}

void Puzzle::erase(const Request& suspect)
{
	std::erase_if(tasks, [&suspect](const Task& t) { return t.request == suspect; });
}

uint64 Puzzle::Nodes() const
{
	return sum(tasks.begin(), tasks.end(),
		0ULL,
		[](const Task& task) { return task.result.nodes; });
}

std::chrono::duration<double> Puzzle::Duration() const
{
	return sum(tasks.begin(), tasks.end(),
		std::chrono::duration<double>{ 0 },
		[](const Task& task) { return task.result.duration; });
}

bool Puzzle::Contains(const Request& suspect) const
{
	return std::find_if(tasks.begin(), tasks.end(),
		[&suspect](const Task& task) { return task.request == suspect; })
		!= tasks.end();
}

Result Puzzle::ResultOf(const Request& request) const noexcept(false)
{
	if (not Contains(request))
		throw std::runtime_error("Puzzle does not contain Request.");
	auto it = std::find_if(tasks.begin(), tasks.end(),
		[&request](const Task& task) { return task.request == request; });
	return it->result;
}

Result Puzzle::ResultOfSecond(const Request& request) const noexcept(false)
{
	if (not Contains(request))
		throw std::runtime_error("Puzzle does not contain Request.");
	auto it = std::find_if(tasks.begin(), tasks.end(),
		[&request](const Task& task) { return task.request == request; });
	it = std::find_if(it + 1, tasks.end(),
		[&request](const Task& task) { return task.request == request; });
	return it->result;
}

bool Puzzle::HasTaskWithoutMove() const
{
	return std::any_of(tasks.begin(), tasks.end(),
		[](const Task& task) { return task.HasMove() == false; });
}

Puzzle::Task Puzzle::MaxIntensity() const noexcept(false)
{
	return MaxIntensity([](const Search::Intensity& l, const Search::Intensity& r) { return l < r; });
}

Puzzle::Task Puzzle::MaxIntensity(const std::function<bool(const Search::Intensity&, const Search::Intensity&)>& less) const noexcept(false)
{
	std::size_t max_element = tasks.size();
	for (std::size_t i = 0; i < tasks.size(); i++)
		if (tasks[i].request.HasMove() == false)
			if (max_element == tasks.size() or less(tasks[max_element].request.intensity, tasks[i].request.intensity))
				max_element = i;
	if (max_element == tasks.size())
		throw std::runtime_error("Puzzle has no task without move.");
	return tasks[max_element];
}

void Puzzle::RemoveAllUndone()
{
	std::erase_if(tasks, [](const Task& task) { return task.IsDone() == false; });
}

void Puzzle::RemoveResult(const Request& request)
{
	auto it = std::find_if(tasks.begin(), tasks.end(),
		[&request](const Task& task) { return task.request == request; });
	if (it != tasks.end())
		it->RemoveResult();
}

void Puzzle::DuplicateRequests()
{
	std::size_t size = tasks.size();
	tasks.reserve(size * 2);
	for (const Task& task : tasks)
		tasks.push_back(task.request);
}

bool Puzzle::Solve(const Search::Algorithm& algorithm)
{
	bool had_work = false;
	for (Task& task : tasks)
	{
		if (task.IsDone())
			continue;

		auto alg = algorithm.Clone();
		auto position = pos;
		if (task.HasMove())
			position = Play(position, task.Move());

		const auto start = std::chrono::high_resolution_clock::now();
		int score = alg->Eval(position, task.Intensity()).Score();
		const auto stop = std::chrono::high_resolution_clock::now();

		task.result = Result(score, alg->nodes, stop - start);
		had_work = true;
	}
	return had_work;
}

std::string Puzzle::to_string() const
{
	// Position
	std::string ret = ::to_string(pos);

	for (const Task& task : tasks)
	{
		bool has_move = task.HasMove();
		auto intensity = task.Intensity();

		ret += " (";
		// move
		if (has_move)
			ret += ::to_string(task.Move()) + " ";
		// score
		ret += DoubleDigitSignedInt(has_move ? -task.result.score : task.result.score);
		// intensity
		if ((not intensity.IsCertain()) or intensity.depth < pos.EmptyCount() - (has_move ? 1 : 0))
			ret += " d" + ::to_string(intensity);
		ret += ")";
	}
	return ret;
}

uint64 Nodes(const std::vector<Puzzle>& vec)
{
	return std::transform_reduce(vec.begin(), vec.end(),
		0ULL, std::plus(),
		[](const Puzzle& p) { return p.Nodes(); });
}

std::chrono::duration<double> Duration(const std::vector<Puzzle>& vec)
{
	return std::transform_reduce(vec.begin(), vec.end(),
		std::chrono::duration<double>(0), std::plus(),
		[](const Puzzle& p) { return p.Duration(); });
}


//uint64 PuzzleProject::Nodes() const
//{
//	std::unique_lock lock(mutex);
//	return std::transform_reduce(wu.begin(), wu.end(),
//		0ULL, std::plus(),
//		[](const auto& p) { return p.Nodes(); });
//}
//
//std::chrono::duration<double> PuzzleProject::Duration() const
//{
//	std::unique_lock lock(mutex);
//	return std::transform_reduce(wu.begin(), wu.end(),
//		std::chrono::duration<double>(0), std::plus(),
//		[](const auto& p) { return p.Duration(); });
//}
//
//void PuzzleProject::MakeAllHave(const ::Request& r)
//{
//	for (Puzzle& p : wu)
//		if (not p.Contains(r))
//			p.push_back(r);
//}
//
//void PuzzleProject::PrepareForTests()
//{
//	for (Puzzle& p : wu)
//		p.PrepareForTests();
//}
//
//uint64 Nodes(const PuzzleProject& proj)
//{
//	return proj.Nodes();
//}
//uint64 Nodes(const std::vector<PuzzleProject>& vec)
//{
//	return std::transform_reduce(vec.begin(), vec.end(),
//		0ULL,
//		std::plus(),
//		[](const auto& p) { return Nodes(p); });
//}
//
//std::chrono::duration<double> Duration(const PuzzleProject& proj)
//{
//	return proj.Duration();
//}
//std::chrono::duration<double> Duration(const std::vector<PuzzleProject>& vec)
//{
//	return std::transform_reduce(vec.begin(), vec.end(),
//		std::chrono::duration<double>(0),
//		std::plus(),
//		[](const auto& p) { return Duration(p); });
//}
//
//void MakeAllHave(PuzzleProject& proj, const ::Request& r)
//{
//	proj.MakeAllHave(r);
//}
//void MakeAllHave(std::vector<PuzzleProject>& vec, const ::Request& r)
//{
//	for (auto& p : vec)
//		MakeAllHave(p, r);
//}
