#include "PositionGenerator.h"
#include "Machine.h"

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
	// Each field has a:
	//  25% chance to belong to player,
	//  25% chance to belong to opponent,
	//  50% chance to be empty.

	auto rnd = [this]() { return BitBoard(std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFULL)(rnd_engine)); };
	BitBoard p = rnd();
	BitBoard o = rnd();
	return { p & ~o, o & ~p };
}

Position PositionGenerator::Random(const std::size_t target_empty_count)
{
	auto dichotron = [this]() { return std::uniform_int_distribution<int>(0, 1)(rnd_engine) == 0; };

	BitBoard P = 0;
	BitBoard O = 0;
	for (std::size_t empty_count = 64; empty_count > target_empty_count; empty_count--)
	{
		auto rnd = std::uniform_int_distribution<std::size_t>(0, empty_count - 1)(rnd_engine);
		auto bit = BitBoard(PDep(1ULL << rnd, Position(P, O).Empties()));

		if (dichotron())
			P |= bit;
		else
			O |= bit;
	}
	return { P, O };
}

Position PositionGenerator::Played(Player& player, std::size_t empty_count, const Position start)
{
	Position pos = start;
	while (pos.EmptyCount() > empty_count)
	{
		try
		{
			pos = player.Play(pos);
		}
		catch (const no_moves_exception&)
		{
			pos = start;
		}
	}
	return pos;
}

//std::vector<Position> PositionGenerator::Played(Player& player, std::size_t size, std::size_t empty_count, Position start)
//{
//	// TODO: Benchmark if unordered_set is faster!
//	//auto hash = [](const Position& pos) { return (pos.GetP() ^ (pos.GetP() >> 36)) * (pos.GetO() ^ (pos.GetO() >> 21)); };
//
//	auto less = [](const Position& l, const Position& r) { return (l.GetP() == r.GetP()) ? (l.GetO() < r.GetO()) : (l.GetP() < r.GetP()); };
//	std::set<Position, decltype(less)> c;
//
//	while (c.size() < size)
//		std::generate_n(std::inserter(c, c.end()), size - c.size(), [&]() { return Played(player, empty_count, start); });
//
//	return { c.begin(), c.end() };
//}

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
//		pos = PlayPass(pos);
//		if (pos.HasMoves())
//			GenAll(pos, pos_set, depth);
//		return;
//	}
//
//	while (!moves.empty())
//	{
//		const auto move = moves.ExtractMove();
//		GenAll(Play(pos, move), pos_set, depth - 1);
//	}
//}
//
//// Taking symmetrie into account.
//void GenerateAllSymmetricUnique(Position pos, std::unordered_set<Position>& pos_set, const uint8_t depth)
//{
//	if (depth == 0) {
//		pos.FlipsToMin();
//		pos_set.insert(pos);
//		return;
//	}
//
//	auto moves = pos.PossibleMoves();
//
//	if (moves.empty())
//	{
//		pos = PlayPass(pos);
//		if (pos.HasMoves())
//			GenAllSym(pos, pos_set, depth);
//		return;
//	   }
//

//	while (!moves.empty())
//	{
//		const auto move = moves.ExtractMove();
//		GenAllSym(Play(pos, move), pos_set, depth - 1);
//	}
//}


//std::vector<Position> PositionGenerator::AllUnique(std::size_t empty_count, Position start)
//{
//	auto less = [](const Position& l, const Position& r) { return (l.GetP() == r.GetP()) ? (l.GetO() < r.GetO()) : (l.GetP() < r.GetP()); };
//	std::set<Position, decltype(less)> set;
//	All(std::inserter(set, set.end()), empty_count, start);
//	return { set.cbegin(), set.cend() };
//}
//
//std::vector<Position> PositionGenerator::AllSymmetricUnique(std::size_t empty_count, Position start)
//{
//	return {};
//}

//Position PositionGenerator::GenerateRandomPosition(uint8_t EmptiesCount)
//{
//	Position pos = Position::Start();
//
//	for (auto plies = pos.EmptyCount() - EmptiesCount; plies > 0; plies--)
//	{
//		Moves moves = pos.PossibleMoves();
//		if (moves.empty())
//		{
//			pos = PlayPass(pos);
//			moves = pos.PossibleMoves();
//			if (moves.empty())
//				return GenerateRandomPosition(EmptiesCount); // Start again.
//		}
//		for (int i = rnd() % moves.size(); i > 0; i--)
//			moves.ExtractMove();
//		pos = Play(pos, moves.ExtractMove());
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
