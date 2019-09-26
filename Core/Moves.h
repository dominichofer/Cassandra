#pragma once
#include "Machine/BitTwiddling.h"

enum Field : uint8_t
{
	A1, B1, C1, D1, E1, F1, G1, H1,
	A2, B2, C2, D2, E2, F2, G2, H2,
	A3, B3, C3, D3, E3, F3, G3, H3,
	A4, B4, C4, D4, E4, F4, G4, H4,
	A5, B5, C5, D5, E5, F5, G5, H5,
	A6, B6, C6, D6, E6, F6, G6, H6,
	A7, B7, C7, D7, E7, F7, G7, H7,
	A8, B8, C8, D8, E8, F8, G8, H8,
	invalid
};

class Moves
{
	uint64_t m_moves = 0;
public:
	Moves() noexcept = default;
	constexpr explicit Moves(uint64_t moves) noexcept : m_moves(moves) {}

	bool operator==(const Moves&) const;
	bool operator!=(const Moves&) const;
	
	std::size_t size() const;
	bool empty() const;

	bool Has(Field) const;
	Field Peek() const;
	void Pop();
	Field Extract();

	void Remove(Field);
	void Remove(uint64_t moves);
	void Filter(uint64_t moves);

	class Iterator
	{
		uint64_t m_moves;
	public:
		explicit Iterator(const Moves& moves) : m_moves(moves.m_moves) {}
		Iterator& operator++() { RemoveLSB(m_moves); return *this; }
		Field operator*() const { return static_cast<Field>(BitScanLSB(m_moves)); }

		bool operator==(const Iterator& o) { return m_moves == o.m_moves; }
		bool operator!=(const Iterator& o) { return m_moves != o.m_moves; }
	};


	Iterator begin() const { return Iterator(*this); }
	Iterator cbegin() const { return Iterator(*this); }
	Iterator end() const { return Iterator(Moves(0)); }
	Iterator cend() const { return Iterator(Moves(0)); }
};

constexpr Moves operator""_mov(const char* c, std::size_t size)
{
	if (size != 64)
		throw "Invalid length of Position string literal";
	uint64_t moves = 0;
	for (int i = 0; i < 64; i++)
		if (c[63 - i] != ' ')
			SetBit(moves, i);
	return Moves(moves);
}