#include "Moves.h"
#include "Machine.h"

bool Moves::operator==(const Moves& o) const
{
	return m_moves == o.m_moves;
}

bool Moves::operator!=(const Moves& o) const
{
	return m_moves != o.m_moves;
}

std::size_t Moves::size() const
{
	return PopCount(m_moves);
}

bool Moves::empty() const
{
	return m_moves == 0;
}

bool Moves::contains(const Field move) const
{
	return m_moves[move];
}

Field Moves::front() const
{
	return static_cast<Field>(BitScanLSB(m_moves));
}

void Moves::pop_front()
{
	RemoveLSB(m_moves);
}

//Field Moves::Extract()
//{
//	const auto LSB = front();
//	pop_front();
//	return LSB;
//}

void Moves::Remove(const Field move)
{
	if (move != Field::invalid) // TODO: Is this needed? Can it be an assert?
		m_moves[move] = false;
}

void Moves::Remove(BitBoard moves)
{
	m_moves &= ~moves;
}

void Moves::Filter(BitBoard moves)
{
	m_moves &= moves;
}