#pragma once
#include "Position.h"
#include "Machine.h"
#include <algorithm>
#include <cstdint>
#include <functional>

class SortedMoves
{
	std::vector<std::pair<int32_t, Field>> m_moves;
public:
	SortedMoves(const Position&, const std::function<int32_t(Field)>& score);

	std::size_t size() const { return m_moves.size(); }
	bool empty() const { return m_moves.empty(); }

	auto begin() const noexcept { return m_moves.rbegin(); }
	auto cbegin() const noexcept { return m_moves.crbegin(); }
	auto end() const noexcept { return m_moves.rend(); }
	auto cend() const noexcept { return m_moves.crend(); }
};

inline SortedMoves::SortedMoves(const Position& pos, const std::function<int32_t(Field)>& score)
{
	Moves moves = PossibleMoves(pos);
	m_moves.reserve(moves.size());

	for (auto move : moves)
		m_moves.emplace_back(score(move), move);

	std::sort(m_moves.begin(), m_moves.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
}
