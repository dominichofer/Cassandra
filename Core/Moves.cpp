#include "Moves.h"
#include "Machine.h"

bool Moves::operator==(const Moves& o) const
{
	return m_moves == o.m_moves;
}

std::size_t Moves::size() const
{
	return PopCount(m_moves);
}

bool Moves::empty() const
{
	return m_moves == 0;
}

void Moves::clear()
{
	m_moves = 0;
}

bool Moves::Has(const Field move) const
{
	return TestBit(m_moves, move);
}

Field Moves::Peek() const
{
	return static_cast<Field>(BitScanLSB(m_moves));
}

Field Moves::Extract()
{
	const auto LSB = static_cast<Field>(BitScanLSB(m_moves));
	RemoveLSB(m_moves);
	return LSB;
}

void Moves::Remove(const Field move)
{
	if (move != Field::invalid)
		ResetBit(m_moves, move);
}

void Moves::Remove(uint64_t moves)
{
	m_moves &= ~moves;
}

void Moves::Filter(uint64_t moves)
{
	m_moves &= moves;
}