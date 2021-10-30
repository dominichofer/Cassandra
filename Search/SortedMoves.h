#pragma once
#include "Core/Core.h"
#include <algorithm>
#include <functional>

class SortedMoves
{
	std::vector<std::pair<int32_t, Field>> m_moves; // TODO: Implement custom Iterator!
public:
	SortedMoves(const Moves& possible_moves, const std::function<int32_t(Field)>& metric)
	{
		m_moves.reserve(possible_moves.size());
		for (const auto& move : possible_moves)
			m_moves.emplace_back(metric(move), move);
		std::sort(m_moves.begin(), m_moves.end(), [](const auto& l, const auto& r) { return l.first > r.first; });
	}

	[[nodiscard]] std::size_t size() const noexcept { return m_moves.size(); }
	[[nodiscard]] bool empty() const noexcept { return m_moves.empty(); }

	[[nodiscard]] decltype(auto) begin() const noexcept { return m_moves.begin(); }
	[[nodiscard]] decltype(auto) cbegin() const noexcept { return m_moves.cbegin(); }
	[[nodiscard]] decltype(auto) end() const noexcept { return m_moves.end(); }
	[[nodiscard]] decltype(auto) cend() const noexcept { return m_moves.cend(); }
};