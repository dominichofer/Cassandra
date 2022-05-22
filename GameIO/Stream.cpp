#include "Stream.h"
#include "SearchIO/SearchIO.h"
#include "CoreIO/CoreIO.h"

void Serialize(const NoMovePuzzle::Task& t, std::ostream& stream)
{
	Serialize(t.request, stream);
	Serialize(t.result, stream);
}

template <>
NoMovePuzzle::Task Deserialize<NoMovePuzzle::Task>(std::istream& stream)
{
	auto request = Deserialize<decltype(NoMovePuzzle::Task::request)>(stream);
	auto result = Deserialize<decltype(NoMovePuzzle::Task::result)>(stream);
	return { request, result };
}

void Serialize(const NoMovePuzzle& p, std::ostream& stream)
{
	Serialize(p.pos, stream);
	Serialize(p.tasks, stream);
}

template <>
NoMovePuzzle Deserialize<NoMovePuzzle>(std::istream& stream)
{
	auto pos = Deserialize<decltype(NoMovePuzzle::pos)>(stream);
	auto tasks = Deserialize<decltype(NoMovePuzzle::tasks)>(stream);
	return { pos, std::move(tasks) };
}

void Serialize(const AllMovePuzzle::Task::SubTask& t, std::ostream& stream)
{
	Serialize(t.move, stream);
	Serialize(t.result, stream);
}

template <>
AllMovePuzzle::Task::SubTask Deserialize<AllMovePuzzle::Task::SubTask>(std::istream& stream)
{
	auto move = Deserialize<decltype(AllMovePuzzle::Task::SubTask::move)>(stream);
	auto result = Deserialize<decltype(AllMovePuzzle::Task::SubTask::result)>(stream);
	return { move, result };
}

void Serialize(const AllMovePuzzle::Task& t, std::ostream& stream)
{
	Serialize(t.request, stream);
	Serialize(t.results, stream);
}

template <>
AllMovePuzzle::Task Deserialize<AllMovePuzzle::Task>(std::istream& stream)
{
	auto request = Deserialize<decltype(AllMovePuzzle::Task::request)>(stream);
	auto results = Deserialize<decltype(AllMovePuzzle::Task::results)>(stream);
	return { request, std::move(results) };
}

void Serialize(const AllMovePuzzle& p, std::ostream& stream)
{
	Serialize(p.pos, stream);
	Serialize(p.tasks, stream);
}

template <>
AllMovePuzzle Deserialize<AllMovePuzzle>(std::istream& stream)
{
	auto pos = Deserialize<decltype(AllMovePuzzle::pos)>(stream);
	auto tasks = Deserialize<decltype(AllMovePuzzle::tasks)>(stream);
	return { pos, std::move(tasks) };
}
