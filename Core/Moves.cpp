#include "Moves.h"
#include "Machine.h"

std::size_t Moves::size() const
{
	return PopCount(m_moves);
}

bool Moves::empty() const noexcept
{
	return m_moves.empty();
}

bool Moves::contains(const Field move) const
{
	return m_moves[move];
}

Field Moves::front() const
{
	return static_cast<Field>(CountTrailingZeros(m_moves));
}

void Moves::pop_front()
{
	m_moves.RemoveFirstField();
}

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