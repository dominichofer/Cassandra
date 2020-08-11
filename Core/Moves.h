#pragma once
#include "BitBoard.h"

class Moves
{
	BitBoard b{};

	class Iterator
	{
		BitBoard moves{};
	public:
		constexpr Iterator() noexcept = default;
		Iterator(const BitBoard& moves) : moves(moves) {}
		[[nodiscard]] Iterator& operator++() { moves.ClearFirstSet(); return *this; }
		[[nodiscard]] Field operator*() const { return moves.FirstSet(); }

		[[nodiscard]] bool operator==(const Iterator& o) const noexcept { return moves == o.moves; }
		[[nodiscard]] bool operator!=(const Iterator& o) const noexcept { return moves != o.moves; }
	};
public:
	constexpr Moves() noexcept = default;
	constexpr Moves(BitBoard moves) noexcept : b(moves) {}

	[[nodiscard]] auto operator<=>(const Moves&) const noexcept = default;
	[[nodiscard]] operator bool() const noexcept { return b; }

	[[nodiscard]] int size() const noexcept { return popcount(b); }
	//[[nodiscard]] constexpr bool empty() const noexcept { return b.empty(); }

	[[nodiscard]] bool contains(Field f) const noexcept { return b.Get(f); }
	[[nodiscard]] Field First() const noexcept { return b.FirstSet(); }
	void RemoveFirst() noexcept { b.ClearFirstSet(); }
	[[nodiscard]] Field ExtractFirst() noexcept { auto first = First(); RemoveFirst(); return first; }

	void Remove(Field f) noexcept { b.Clear(f); }
	void Remove(const BitBoard& moves) noexcept { b &= ~moves; }
	void Filter(const BitBoard& moves) noexcept { b &= moves; }

	[[nodiscard]] Iterator begin() const { return Iterator(b); }
	[[nodiscard]] Iterator cbegin() const { return Iterator(b); }
	[[nodiscard]] static Iterator end() { return {}; }
	[[nodiscard]] static Iterator cend() { return {}; }
};
