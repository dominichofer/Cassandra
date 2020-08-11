#pragma once
#include "Core/Core.h"
#include <algorithm>
#include <cstdint>
#include <functional>

class SortedMoves
{
	std::vector<std::pair<int32_t, Field>> m_moves;
public:
	SortedMoves(const Moves&, const std::function<int32_t(Field)>& score);

	[[nodiscard]] std::size_t size() const noexcept { return m_moves.size(); }
	[[nodiscard]] bool empty() const noexcept { return m_moves.empty(); }

	[[nodiscard]] auto begin() const noexcept { return m_moves.rbegin(); }
	[[nodiscard]] auto cbegin() const noexcept { return m_moves.crbegin(); }
	[[nodiscard]] auto end() const noexcept { return m_moves.rend(); }
	[[nodiscard]] auto cend() const noexcept { return m_moves.crend(); }
};

inline SortedMoves::SortedMoves(const Moves& possible_moves, const std::function<int32_t(Field)>& score)
{
	m_moves.reserve(possible_moves.size());

	for (const auto& move : possible_moves)
		m_moves.emplace_back(score(move), move);

	std::sort(m_moves.begin(), m_moves.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
}
