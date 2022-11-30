#pragma once
#include "BitBoard.h"
#include <iterator>
#include <ranges>

namespace moves
{
	class Iterator
	{
		BitBoard moves;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		CUDA_CALLABLE Iterator() noexcept = default;
		CUDA_CALLABLE Iterator(const BitBoard& moves) noexcept : moves(moves) {}

		CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return moves == o.moves; }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return moves != o.moves; }

		CUDA_CALLABLE Iterator& operator++() noexcept { moves.ClearFirstSet(); return *this; }
		CUDA_CALLABLE Field operator*() const noexcept { return moves.FirstSetField(); }
	};
}

class Moves
{
	BitBoard b;
public:
	constexpr Moves() noexcept = default;
	CUDA_CALLABLE constexpr explicit Moves(BitBoard moves) noexcept : b(moves) {}

	CUDA_CALLABLE bool operator==(const Moves& o) const noexcept { return b == o.b; }
	CUDA_CALLABLE bool operator!=(const Moves& o) const noexcept { return b != o.b; }
	CUDA_CALLABLE Moves operator&(const BitBoard& mask) const noexcept { return Moves{ b & mask }; }

	CUDA_CALLABLE Field operator[](std::size_t index) const noexcept { return BitBoard(PDep(1ULL << index, b)).FirstSetField(); }

	CUDA_CALLABLE operator bool() const noexcept { return b; }
	CUDA_CALLABLE operator BitBoard() const noexcept { return b; }

	CUDA_CALLABLE bool empty() const noexcept { return b.empty(); }
	CUDA_CALLABLE bool contains(Field move) const noexcept { return b.Get(move); }
	CUDA_CALLABLE void erase(Field move) noexcept { return b.Clear(move); }
	CUDA_CALLABLE std::size_t size() const noexcept { return popcount(b); }

	CUDA_CALLABLE Field front() const noexcept { return b.FirstSetField(); }
	CUDA_CALLABLE void pop_front() noexcept { b.ClearFirstSet(); }

	CUDA_CALLABLE moves::Iterator begin() const noexcept { return b; }
	CUDA_CALLABLE moves::Iterator end() const noexcept { return {}; }
};
