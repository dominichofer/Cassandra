#include "Interval.h"

OpenInterval::OpenInterval(ClosedInterval o) noexcept
	: OpenInterval(o.m_lower - 1, o.m_upper + 1)
{}

[[nodiscard]]
OpenInterval& OpenInterval::operator-=(ClosedInterval o) noexcept
{
	assert(Contains(o.lower()) != Contains(o.upper()));

	if (Contains(o.lower()))
		m_upper = o.lower();
	else if (Contains(o.upper()))
		m_lower = o.upper();
	return *this;
}

[[nodiscard]] 
ClosedInterval& ClosedInterval::operator-=(OpenInterval o) noexcept
{
	assert(Contains(o.lower()) != Contains(o.upper()));

	if (Contains(o.lower()))
		m_upper = o.lower();
	else if (Contains(o.upper()))
		m_lower = o.upper();
	return *this;
}


[[nodiscard]] bool OpenInterval::Contains(ClosedInterval o) const noexcept
{
	return (m_lower < o.m_lower) && (o.m_upper < m_upper);
}

[[nodiscard]]
bool OpenInterval::Overlaps(OpenInterval o) const noexcept
{
	return !(o < *this) && !(*this < o);
}

[[nodiscard]]
bool OpenInterval::Overlaps(ClosedInterval o) const noexcept
{
	return !(o < *this) && !(*this < o);
}

[[nodiscard]]
bool ClosedInterval::Overlaps(OpenInterval o) const noexcept
{
	return !(o < *this) && !(*this < o);
}

[[nodiscard]]
bool ClosedInterval::Overlaps(ClosedInterval o) const noexcept
{
	return !(o < *this) && !(*this < o);
}

[[nodiscard]] OpenInterval Overlap(OpenInterval l, OpenInterval r) noexcept
{
	assert(l.Overlaps(r));
	return { std::max(l.m_lower, r.m_lower), std::min(l.m_upper, r.m_upper) };
}

[[nodiscard]] ClosedInterval Overlap(ClosedInterval l, ClosedInterval r) noexcept
{
	assert(l.Overlaps(r));
	return { std::max(l.m_lower, r.m_lower), std::min(l.m_upper, r.m_upper) };
}

[[nodiscard]] OpenInterval Hull(OpenInterval l, OpenInterval r) noexcept
{
	return { std::min(l.m_lower, r.m_lower), std::max(l.m_upper, r.m_upper) };
}

[[nodiscard]] ClosedInterval Hull(ClosedInterval l, ClosedInterval r) noexcept
{
	return { std::min(l.m_lower, r.m_lower), std::max(l.m_upper, r.m_upper) };
}
