#include "Perft.h"
#include "Hashtable.h"
#include "Core/Core.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <optional>
#include <unordered_map>
#include <omp.h>

std::size_t Correct(int depth)
{
	const std::size_t correct[] =
	{
		                     1,
		                     4,
		                    12,
		                    56,
		                   244,
		                 1'396,
		                 8'200,
		                55'092,
		               390'216,
		             3'005'288,
		            24'571'056,
		           212'258'216,
				 1'939'879'668,
				18'429'618'408,
			   184'041'761'768,
			 1'891'831'332'208,
			20'301'171'282'452,
		   222'742'563'853'912,
		 2'534'535'926'617'852,
		29'335'558'770'589'276,
		                    20,
		                    21,
		                    22,
		                    23,
		                    24,
		                    25,
	};
	return correct[depth];
}

namespace Basic
{
	std::size_t perft(Position pos, const int depth)
	{
		if (depth == 0)
			return 1;

		auto moves = PossibleMoves(pos);

		if (moves.empty())
		{
			Position passed = PlayPass(pos);
			if (PossibleMoves(passed).empty())
				return 0;
			return perft(passed, depth - 1);
		}

		std::size_t sum = 0;
		for (auto move : moves)
			sum += perft(Play(pos, move), depth - 1);

		return sum;
	}

	std::size_t perft(const int depth)
	{
		Position pos = Position::Start();

		if (depth == 0)
			return perft(pos, depth);

		// Makes use of 4-fold symmetrie.
		pos = Play(pos, PossibleMoves(pos).First());
		return 4 * perft(pos, depth - 1);
	}
}

namespace Unrolled2
{
	// perft for 0 plies left
	std::size_t perft_0(Position)
	{
		return 1;
	}

	// perft for 1 ply left
	std::size_t perft_1(Position pos)
	{
		return PossibleMoves(pos).size();
	}

	// perft for 2 plies left
	std::size_t perft_2(Position pos)
	{
		auto moves = PossibleMoves(pos);

		if (moves.empty())
			return PossibleMoves(PlayPass(pos)).size();

		std::size_t sum = 0;
		for (auto move : moves)
		{
			const auto next_pos = Play(pos, move);
			const auto next_moves = PossibleMoves(next_pos);
			if (next_moves.empty())
				sum += static_cast<std::size_t>(PossibleMoves(PlayPass(next_pos)).empty() ? 0 : 1);
			else
				sum += next_moves.size();
		}

		return sum;
	}

	std::size_t perft_(Position pos, const int depth)
	{
		if (depth == 2)
			return perft_2(pos);

		auto moves = PossibleMoves(pos);

		if (moves.empty())
		{
			Position passed = PlayPass(pos);
			if (PossibleMoves(passed).empty())
				return 0;
			return perft_(passed, depth - 1);
		}

		std::size_t sum = 0;
		for (auto move : moves)
			sum += perft_(Play(pos, move), depth - 1);

		return sum;
	}

	std::size_t perft(Position pos, const int depth)
	{
		switch (depth)
		{
			case 0: return perft_0(pos);
			case 1: return perft_1(pos);
			case 2: return perft_2(pos);
			default: return perft_(pos, depth);
		}
	}

	std::size_t perft(const int depth)
	{
		Position pos = Position::Start();

		if (depth == 0)
			return perft(pos, depth);

		// Makes use of 4-fold symmetrie.
		pos = Play(pos, PossibleMoves(pos).First());
		return 4 * perft(pos, depth - 1);
	}
}

namespace HashTableMap
{
	bool operator<(Position l, Position r) noexcept { return (l.P == r.P) ? (l.O < r.O) : (l.P < r.P); }

	void fill(Position pos, const uint8_t depth, std::vector<Position>& all)
	{
		if (depth == 0)
		{
			all.push_back(FlipToUnique(pos));
			return;
		}

		auto moves = PossibleMoves(pos);

		if (moves.empty())
		{
			auto passed = PlayPass(pos);
			if (!PossibleMoves(passed).empty())
				fill(passed, depth - 1, all);
			return;
		}

		for (auto move : moves)
			fill(Play(pos, move), depth - 1, all);
	}

	struct PositionDegeneracy
	{
		Position pos;
		std::size_t degeneracy;
	};

	std::vector<PositionDegeneracy> GetWork(Position pos, const std::size_t uniqueness_depth)
	{
		std::vector<Position> all;
		fill(pos, uniqueness_depth, all);
		std::sort(all.begin(), all.end(), operator<);

		std::vector<PositionDegeneracy> work;
		work.push_back({ pos, 0 });
		for (const auto& pos : all)
		{
			if (work.back().pos == pos)
				work.back().degeneracy++;
			else
				work.push_back({ pos, 1 });
		}
		return work;
	}

	class PerftWithHashTable
	{
		BigNodeHashTable hash_table;

		std::size_t perft(Position, int depth);
	public:
		PerftWithHashTable(std::size_t BytesRAM) : hash_table(BytesRAM / sizeof(BigNodeHashTable::node_type)) {}
		std::size_t calculate(Position, int depth);
	};

	std::size_t PerftWithHashTable::perft(Position pos, const int depth)
	{
		if (depth == 2)
			return Unrolled2::perft_2(pos);

		auto moves = PossibleMoves(pos);

		if (moves.empty())
		{
			Position passed = PlayPass(pos);
			if (PossibleMoves(passed).empty())
				return 0;
			return perft(passed, depth - 1);
		}

		if (const auto ret = hash_table.LookUp({ pos, depth }); ret.has_value())
			return ret.value();

		std::size_t sum = 0;
		for (auto move : moves)
			sum += perft(Play(pos, move), depth - 1);

		hash_table.Update({ pos, depth }, sum);
		return sum;
	}

	std::size_t PerftWithHashTable::calculate(Position pos, const int depth)
	{
		const std::size_t uniqueness_depth = 9;
		std::vector<PositionDegeneracy> work = GetWork(pos, uniqueness_depth);
		const auto size = static_cast<int64_t>(work.size());

		std::size_t sum = 0;
		#pragma omp parallel for schedule(dynamic,1) reduction(+:sum)
		for (int64_t i = 0; i < size; i++)
			sum += perft(work[i].pos, depth - uniqueness_depth) * work[i].degeneracy;
		return sum;
	}
	
	std::size_t perft(Position pos, int depth, std::size_t BytesRAM)
	{
		return PerftWithHashTable(BytesRAM).calculate(pos, depth - 1);
	}

	std::size_t perft(const int depth, const std::size_t BytesRAM)
	{
		Position pos = Position::Start();

		if (depth < 13)
			return Unrolled2::perft(pos, depth);

		// Makes use of 4-fold symmetrie.
		pos = Play(pos, PossibleMoves(pos).First());
		return 4 * perft(pos, depth, BytesRAM);
	}
}
