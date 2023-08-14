#pragma once
#include "Board/Board.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <vector>

class SortedMoves
{
	std::vector<uint32_t> m_moves;
public:
	class Iterator
	{
		std::vector<uint32_t>::const_iterator it;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		CUDA_CALLABLE Iterator(std::vector<uint32_t>::const_iterator it) noexcept : it(it) {}

		CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return it == o.it; }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return it != o.it; }

		CUDA_CALLABLE Iterator& operator++() noexcept { ++it; return *this; }
		CUDA_CALLABLE Field operator*() const noexcept { return static_cast<Field>(*it); }
	};

	SortedMoves(const Moves& possible_moves, std::function<uint32_t(Field)> metric) noexcept
	{
		m_moves.reserve(possible_moves.size());
		for (Field move : possible_moves)
			m_moves.push_back((metric(move) << 8) + std::to_underlying(move));
		std::ranges::sort(m_moves, std::greater<uint32_t>());
	}

	Field operator[](std::size_t index) const noexcept { return static_cast<Field>(m_moves[index]); }

	std::size_t size() const noexcept { return m_moves.size(); }
	bool empty() const noexcept { return m_moves.empty(); }

	CUDA_CALLABLE Iterator begin() const noexcept { return m_moves.begin(); }
	CUDA_CALLABLE Iterator end() const noexcept { return m_moves.end(); }
};
