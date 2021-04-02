#include "Interval.h"

OpenInterval::OpenInterval(ClosedInterval o) noexcept
	: OpenInterval(o.lower() - 1, o.upper() + 1)
{}

OpenInterval& OpenInterval::operator-=(ClosedInterval o) noexcept
{
	assert(Contains(o.lower()) != Contains(o.upper()));

	if (Contains(o.lower()))
		m_upper = o.lower();
	else if (Contains(o.upper()))
		m_lower = o.upper();
	return *this;
}

ClosedInterval& ClosedInterval::operator-=(OpenInterval o) noexcept
{
	assert(Contains(o.lower()) != Contains(o.upper()));

	if (Contains(o.lower()))
		m_upper = o.lower();
	else if (Contains(o.upper()))
		m_lower = o.upper();
	return *this;
}

bool OpenInterval::Contains(ClosedInterval o) const noexcept
{
	return (m_lower < o.lower()) && (o.upper() < m_upper);
}

bool OpenInterval::Overlaps(ClosedInterval o) const noexcept
{
	return (m_lower < o.upper()) && (m_upper > o.lower());
}

OpenInterval Overlap(OpenInterval l, OpenInterval r) noexcept
{
	assert(l.Overlaps(r));
	return { std::max(l.lower(), r.lower()), std::min(l.upper(), r.upper()) };
}

ClosedInterval Overlap(ClosedInterval l, ClosedInterval r) noexcept
{
	assert(l.Overlaps(r));
	return { std::max(l.lower(), r.lower()), std::min(l.upper(), r.upper()) };
}

OpenInterval Hull(OpenInterval l, OpenInterval r) noexcept
{
	return { std::min(l.lower(), r.lower()), std::max(l.upper(), r.upper()) };
}

ClosedInterval Hull(ClosedInterval l, ClosedInterval r) noexcept
{
	return { std::min(l.lower(), r.lower()), std::max(l.upper(), r.upper()) };
}
