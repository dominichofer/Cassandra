#pragma once
#include "Core/Core.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <tuple>
#include <iterator>
#include <valarray>

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

		constexpr Iterator() noexcept = default;
		CUDA_CALLABLE Iterator(std::vector<uint32_t>::const_iterator it) noexcept : it(it) {}
		CUDA_CALLABLE Iterator& operator++() noexcept { ++it; return *this; }
		CUDA_CALLABLE Field operator*() const noexcept { return static_cast<Field>(*it); }

		CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return it == o.it; }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return it != o.it; }
	};

	SortedMoves(const Moves& possible_moves, auto&& metric)
	{
		m_moves.reserve(possible_moves.size());
		for (Field move : possible_moves)
			m_moves.push_back(metric(move) + static_cast<uint8_t>(move));
		std::ranges::sort(m_moves);
	}

	std::size_t size() const noexcept { return m_moves.size(); }
	bool empty() const noexcept { return m_moves.empty(); }

	CUDA_CALLABLE Iterator begin() const noexcept { return m_moves.begin(); }
	CUDA_CALLABLE Iterator end() const noexcept { return m_moves.end(); }
};


inline BitBoard PotentialMoves(const Position& pos) noexcept
{
	return EightNeighboursAndSelf(pos.Opponent()) & pos.Empties();
}

inline CUDA_CALLABLE int DoubleCornerPopcount(const BitBoard& b) noexcept { return popcount(b) + popcount(b & BitBoard::Corners()); }
inline CUDA_CALLABLE int DoubleCornerPopcount(const Moves& m) noexcept { return m.size() + (m & BitBoard::Corners()).size(); }

class MoveSorter
{
	// Factory of SortedMoves

	std::valarray<float> weights;
public:
	MoveSorter(std::valarray<float> weights) noexcept : weights(weights) {}
	//MoveSorter() : MoveSorter(5, 6, 15, 11) {}
	
	SortedMoves Sort(const Moves&, const Position&) const;
};
