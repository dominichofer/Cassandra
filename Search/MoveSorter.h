#pragma once
#include "Board/Board.h"
#include "HashTable.h"
#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

// Forward declarations
class Algorithm;

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

		CUDA_CALLABLE bool operator==(const Iterator& o) const { return it == o.it; }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const { return it != o.it; }

		CUDA_CALLABLE Iterator& operator++() { ++it; return *this; }
		CUDA_CALLABLE Field operator*() const { return static_cast<Field>(*it); }
	};

	SortedMoves(Moves possible_moves, std::function<uint32_t(Field)> metric);

	Field operator[](std::size_t index) const { return static_cast<Field>(m_moves[index]); }

	std::size_t size() const { return m_moves.size(); }
	bool empty() const { return m_moves.empty(); }

	CUDA_CALLABLE Iterator begin() const { return m_moves.begin(); }
	CUDA_CALLABLE Iterator end() const { return m_moves.end(); }
};


class MoveSorter
{
	HashTable& tt;
	Algorithm& alg;
public:
	MoveSorter(HashTable& tt, Algorithm& alg) noexcept : tt(tt), alg(alg) {}

	SortedMoves Sorted(const Position&, Intensity);
};