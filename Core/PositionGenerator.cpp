#include "PositionGenerator.h"
#include "Machine.h"

#include <stack>
#include <set>
#include <functional>
#include <tuple>
#include <iterator>

// TODO: Remove!
//#include <algorithm>
//#include <atomic>
//#include <functional>
//#include <mutex>
//#include <omp.h>
//#include <thread>

//class ThreadSavePosSet
//{
//	mutable std::mutex mtx;
//	std::unordered_set<Position> set;
//public:
//	bool TryInsert(const Position& pos, std::size_t maxSize);
//	std::size_t size() const;
//	const std::unordered_set<Position>& GetSet() const { return set; }
//	      std::unordered_set<Position>  GetSet()       { return set; }
//};
//
//bool ThreadSavePosSet::TryInsert(const Position& pos, std::size_t maxSize)
//{
//	std::lock_guard<std::mutex> guard(mtx);
//	if (set.size() >= maxSize)
//		return false;
//	const bool InsertionTookPlace = set.insert(pos).second;
//	return InsertionTookPlace;
//}
//
//std::size_t ThreadSavePosSet::size() const
//{
//	std::lock_guard<std::mutex> guard(mtx);
//	return set.size();
//}

Position PositionGenerator::Random()
{
	// middle fields
	// 50% chance to belong to player
	// 50% chance to belong to opponent

	// non-middle fields
	// 50% chance to be empty
	// 25% chance to belong to player
	// 25% chance to belong to opponent

	auto rnd = [this]() { return std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFui64)(rnd_engine); };
	uint64_t p = rnd();
	uint64_t o = rnd();
	Board middle = RandomMiddle();
	return { (p & ~o) | middle.P, (o & ~p) | middle.O };
}

Position PositionGenerator::Random(uint64_t empty_count)
{
	auto dichotron = [this]() { return std::uniform_int_distribution<int>(0, 1)(rnd_engine) == 0; };

	Board board = RandomMiddle();
	while (board.EmptyCount() > empty_count)
	{
		auto rnd = std::uniform_int_distribution<std::size_t>(0, board.EmptyCount())(rnd_engine);
		auto bit = PDep(Bit(rnd), board.Empties());

		if (dichotron())
			board.P |= bit;
		else
			board.O |= bit;
	}
	return board;
}

Position PositionGenerator::Played(Player& player, std::size_t empty_count, Position start)
{
	Position pos = start;
	while (pos.EmptyCount() > empty_count)
	{
		try
		{
			player.Play(pos);
		}
		catch (const no_moves_exception&)
		{
			pos = start;
		}
	}
	return pos;
}

std::vector<Position> PositionGenerator::Played(Player& player, std::size_t size, std::size_t empty_count, Position start)
{
	// TODO: Benchmark if unordered_set is faster!
	//auto hash = [](const Position& pos) { return (pos.GetP() ^ (pos.GetP() >> 36)) * (pos.GetO() ^ (pos.GetO() >> 21)); };

	auto less = [](const Position& l, const Position& r) { return (l.GetP() == r.GetP()) ? (l.GetO() < r.GetO()) : (l.GetP() < r.GetP()); };
	std::set<Position, decltype(less)> c;

	while (c.size() < size)
		std::generate_n(std::inserter(c, c.end()), size - c.size(), [&]() { return Played(player, empty_count, start); });

	return { c.begin(), c.end() };
}

// Not taking symmetrie into account.
//void GenerateAll(Position pos, std::unordered_set<Position>& pos_set, const uint8_t depth)
//{
//	if (depth == 0) {
//		pos_set.insert(pos);
//		return;
//	}
//
//	auto moves = pos.PossibleMoves();
//
//	if (moves.empty())
//	{
//		pos = pos.PlayPass();
//		if (pos.HasMoves())
//			GenAll(pos, pos_set, depth);
//		return;
//	}
//
//	while (!moves.empty())
//	{
//		const auto move = moves.ExtractMove();
//		GenAll(pos.Play(move), pos_set, depth - 1);
//	}
//}
//
//// Taking symmetrie into account.
//void GenerateAllSymmetricUnique(Position pos, std::unordered_set<Position>& pos_set, const uint8_t depth)
//{
//	if (depth == 0) {
//		pos.FlipToMin();
//		pos_set.insert(pos);
//		return;
//	}
//
//	auto moves = pos.PossibleMoves();
//
//	if (moves.empty())
//	{
//		pos = pos.PlayPass();
//		if (pos.HasMoves())
//			GenAllSym(pos, pos_set, depth);
//		return;
//	   }
//

//	while (!moves.empty())
//	{
//		const auto move = moves.ExtractMove();
//		GenAllSym(pos.Play(move), pos_set, depth - 1);
//	}
//}

std::vector<Position> PositionGenerator::All(std::size_t empty_count, Position start)
{
	// TODO: Benchmark to see if unordered_set is faster!

	auto less = [](const Position& l, const Position& r) { return (l.GetP() == r.GetP()) ? (l.GetO() < r.GetO()) : (l.GetP() < r.GetP()); };
	std::set<Position, decltype(less)> c;

	struct pair { Position pos; Moves moves; };
	std::stack<pair> stack;

	stack.push({ start, PossibleMoves(start) });
	while (!stack.empty())
	{
		auto& top = stack.top();
		if (top.moves.empty())
			stack.pop();
		else
		{
			Position new_pos = Play(top.pos, top.moves.Extract());
			if (new_pos.EmptyCount() == empty_count)
				c.insert(new_pos);
			else
				stack.push({ new_pos, PossibleMoves(new_pos) });
		}
	}
	return { c.begin(), c.end() };
}

std::vector<Position> PositionGenerator::AllSymmetricUnique(std::size_t empty_count, Position start)
{
	return {};
}

//Position PositionGenerator::GenerateRandomPosition(uint8_t EmptiesCount)
//{
//	Position pos = Position::Start();
//
//	for (auto plies = pos.EmptyCount() - EmptiesCount; plies > 0; plies--)
//	{
//		Moves moves = pos.PossibleMoves();
//		if (moves.empty())
//		{
//			pos = pos.PlayPass();
//			moves = pos.PossibleMoves();
//			if (moves.empty())
//				return GenerateRandomPosition(EmptiesCount); // Start again.
//		}
//		for (int i = rnd() % moves.size(); i > 0; i--)
//			moves.ExtractMove();
//		pos = pos.Play(moves.ExtractMove());
//	}
//
//	return pos;
//}
//
//std::unordered_set<Position> PositionGenerator::GenerateRandomPositionSet(uint8_t EmptiesCount, std::size_t size)
//{
//	ThreadSavePosSet PosSet;
//	auto gen = [&] { while (PosSet.size() < size) PosSet.TryInsert(GenerateRandomPosition(EmptiesCount), size); };
//
//	std::vector<std::thread> threads;
//	for (std::size_t i = 0; i < std::thread::hardware_concurrency() - 1; i++)
//		threads.emplace_back(gen);
//	gen();
//
//	for (auto& it : threads)
//		it.join();
//
//	return PosSet.GetSet();
//}
//
//std::unordered_set<Position> PositionGenerator::GenerateAllPositions(uint8_t EmptiesCount)
//{
//	std::unordered_set<Position> positions;
//	Position pos = Position::StartPosition();
//	GenAll(pos, positions, static_cast<uint8_t>(pos.EmptyCount() - EmptiesCount));
//	return positions;
//}
//
//std::unordered_set<Position> PositionGenerator::GenerateAllPositionsSym(uint8_t EmptiesCount)
//{
//	std::unordered_set<Position> positions;
//	Position pos = Position::StartPosition();
//	GenAllSym(pos, positions, static_cast<uint8_t>(pos.EmptyCount() - EmptiesCount));
//	return positions;
//}

//std::unordered_set<Position> PositionGenerator::RandomlyPlayed(std::size_t count, Position start_pos)
//{
//	return std::unordered_set<Position>();
//}
//
//std::unordered_set<Position> PositionGenerator::RandomlyPlayed(std::size_t count, uint64_t empty_count, Position start_pos)
//{
//	return std::unordered_set<Position>();
//}
//
//std::unordered_set<Position> PositionGenerator::RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count, Position start_pos)
//{
//	return std::unordered_set<Position>();
//}
//
//std::unordered_set<Position> PositionGenerator::RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count, uint64_t empty_count, Position start_pos)
//{
//	return std::unordered_set<Position>();
//}
//
//std::unordered_set<Position> PositionGenerator::RandomlyPlayed(std::execution::parallel_policy&&, std::size_t count, Position start_pos)
//{
//	return std::unordered_set<Position>();
//}
//
//std::unordered_set<Position> PositionGenerator::RandomlyPlayed(std::execution::parallel_policy&&, std::size_t count, uint64_t empty_count, Position start_pos)
//{
//	return std::unordered_set<Position>();
//}

Board PositionGenerator::RandomMiddle()
{
	uint64_t rnd = std::uniform_int_distribution<uint64_t>(0ui64, ~0ui64)(rnd_engine);
	return { Board::middle & rnd, Board::middle & ~rnd };
}
