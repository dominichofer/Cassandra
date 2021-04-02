#pragma once
#include "BitBoard.h"
#include <iterator>

class Moves
{
	BitBoard b = 0;

	class Iterator
	{
		BitBoard moves{};
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		constexpr Iterator() noexcept = default;
		CUDA_CALLABLE Iterator(const BitBoard& moves) : moves(moves) {}
		CUDA_CALLABLE Iterator& operator++() { moves.ClearFirstSet(); return *this; }
		[[nodiscard]] CUDA_CALLABLE Field operator*() const { return moves.FirstSetField(); }

		[[nodiscard]] CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return moves == o.moves; }
		[[nodiscard]] CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return moves != o.moves; }
	};
public:
	constexpr Moves() noexcept = default;
	CUDA_CALLABLE constexpr Moves(BitBoard moves) noexcept : b(moves) {}

	[[nodiscard]] CUDA_CALLABLE bool operator==(const Moves& o) const noexcept { return b == o.b; }
	[[nodiscard]] CUDA_CALLABLE bool operator!=(const Moves& o) const noexcept { return b != o.b; }

	[[nodiscard]] CUDA_CALLABLE operator bool() const noexcept { return b; }

	[[nodiscard]] CUDA_CALLABLE bool empty() const noexcept { return !b; }
	[[nodiscard]] CUDA_CALLABLE int size() const noexcept { return popcount(b); }
	[[nodiscard]] CUDA_CALLABLE bool contains(Field f) const noexcept { return b.Get(f); }

	[[nodiscard]] CUDA_CALLABLE Field First() const noexcept { return b.FirstSetField(); }
	CUDA_CALLABLE void RemoveFirst() noexcept { b.ClearFirstSet(); }
	[[nodiscard]] CUDA_CALLABLE Field ExtractFirst() noexcept { auto first = First(); RemoveFirst(); return first; }

	CUDA_CALLABLE void Remove(Field move) noexcept { b.Clear(move); }
	CUDA_CALLABLE void Remove(const Moves& moves) noexcept { b &= ~moves.b; }
	CUDA_CALLABLE void Filter(const Moves& moves) noexcept { b &= moves.b; }
	[[nodiscard]] CUDA_CALLABLE Moves Filtered(const Moves& moves) const noexcept { return b & moves.b; }

	[[nodiscard]] CUDA_CALLABLE Iterator begin() const { return b; }
	[[nodiscard]] CUDA_CALLABLE Iterator cbegin() const { return b; }
	[[nodiscard]] CUDA_CALLABLE static Iterator end() { return {}; }
	[[nodiscard]] CUDA_CALLABLE static Iterator cend() { return {}; }
};
