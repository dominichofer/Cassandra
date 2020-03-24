#include "Moves.h"
#include "Machine.h"

std::size_t Moves::size() const
{
	return PopCount(m_moves);
}

bool Moves::empty() const noexcept
{
	return m_moves == 0;
}

Moves::operator bool() const noexcept
{
	return m_moves != 0;
}

bool Moves::contains(const Field move) const
{
	return Bit(m_moves, move);
}

Field Moves::front() const
{
	return static_cast<Field>(CountTrailingZeros(m_moves));
}

Field Moves::pop_front()
{
	auto ret = front();
	RemoveLSB(m_moves);
	return ret;
}

void Moves::Remove(const Field move)
{
	Bit(m_moves, move) = false;
}

void Moves::Remove(BitBoard moves)
{
	m_moves &= ~moves;
}

void Moves::Filter(BitBoard moves)
{
	m_moves &= moves;
}


Moves::Iterator& Moves::Iterator::operator++()
{
	RemoveLSB(m_moves);
	return *this;
}

Field Moves::Iterator::operator*() const
{
	return static_cast<Field>(BitScanLSB(m_moves));
}