#pragma once
#include "BitBoard.h"

class Moves
{
	BitBoard m_moves{ 0 };
public:
	Moves() noexcept = default;
	constexpr explicit Moves(BitBoard moves) noexcept : m_moves(moves) {}

	[[nodiscard]] auto operator<=>(const Moves&) const noexcept = default;
	
	std::size_t size() const;
	bool empty() const noexcept;

	bool contains(Field) const;
	Field front() const;
	void pop_front();

	void Remove(Field);
	void Remove(BitBoard moves);
	void Filter(BitBoard moves);

	class Iterator
	{
		BitBoard m_moves;
	public:
		explicit Iterator(const Moves& moves) : m_moves(moves.m_moves) {}
		Iterator& operator++() { m_moves.RemoveFirstField(); return *this; }
		Field operator*() const { return m_moves.FirstField(); }

		bool operator==(const Iterator& o) { return m_moves == o.m_moves; }
		bool operator!=(const Iterator& o) { return m_moves != o.m_moves; }
	};


	Iterator begin() const { return Iterator(*this); }
	Iterator cbegin() const { return Iterator(*this); }
	Iterator end() const { return Iterator(Moves(BitBoard(0))); }
	Iterator cend() const { return Iterator(Moves(BitBoard(0))); }
};

constexpr Moves operator""_mov(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard moves{ 0 };
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
		{
			char symbol = c[119 - 2 * j - 15 * i];
			if (symbol == '#')
				moves[i * 8 + j] = true;
		}
	return Moves(moves);
}