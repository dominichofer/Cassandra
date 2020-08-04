#pragma once
#include "BitBoard.h"

class Moves
{
	BitBoard b{};
public:
	constexpr Moves() noexcept = default;
	constexpr Moves(BitBoard moves) noexcept : b(moves) {}

	[[nodiscard]] auto operator<=>(const Moves&) const noexcept = default;

	int size() const noexcept { return popcount(b); }
	constexpr bool empty() const noexcept { return b.empty(); }

	bool contains(Field f) const noexcept { return b.Get(f); }
	Field First() const noexcept { return b.FirstSet(); }
	void RemoveFirst() noexcept { b.ClearFirstSet(); }
	Field ExtractFirst() noexcept { auto first = First(); RemoveFirst(); return first; }

	void Remove(Field f) noexcept { b.Clear(f); }
	void Remove(const BitBoard& moves) { b &= ~moves; }
	void Filter(const BitBoard& moves) { b &= moves; }

	class Iterator
	{
		BitBoard moves{};
	public:
		constexpr Iterator() noexcept = default;
		Iterator(const Moves& moves) : moves(moves.b) {}
		Iterator& operator++() { moves.ClearFirstSet(); return *this; }
		Field operator*() const { return moves.FirstSet(); }

		[[nodiscard]] bool operator==(const Iterator& o) const noexcept { return moves == o.moves; }
		[[nodiscard]] bool operator!=(const Iterator& o) const noexcept { return moves != o.moves; }
	};

	Iterator begin() const { return *this; }
	Iterator cbegin() const { return *this; }
	Iterator end() const { return {}; }
	Iterator cend() const { return {}; }
};
