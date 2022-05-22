#include "Interval.h"
#include <cassert>

OpenInterval::OpenInterval(ClosedInterval o) noexcept
	: OpenInterval(o.Lower() - 1, o.Upper() + 1)
{}

OpenInterval& OpenInterval::operator-=(ClosedInterval o) noexcept
{
	assert(Contains(o.Lower()) != Contains(o.Upper()));

	if (Contains(o.Lower()))
		upper = o.Lower();
	else if (Contains(o.Upper()))
		lower = o.Upper();
	return *this;
}

ClosedInterval& ClosedInterval::operator-=(OpenInterval o) noexcept
{
	assert(Contains(o.Lower()) != Contains(o.Upper()));

	if (Contains(o.Lower()))
		upper = o.Lower();
	else if (Contains(o.Upper()))
		lower = o.Upper();
	return *this;
}

bool OpenInterval::Contains(ClosedInterval o) const noexcept
{
	return (lower < o.Lower()) && (o.Upper() < upper);
}

bool OpenInterval::Overlaps(ClosedInterval o) const noexcept
{
	return (lower < o.Upper()) && (upper > o.Lower());
}

OpenInterval Overlap(OpenInterval l, OpenInterval r) noexcept
{
	assert(l.Overlaps(r));
	return { std::max(l.Lower(), r.Lower()), std::min(l.Upper(), r.Upper()) };
}

ClosedInterval Overlap(ClosedInterval l, ClosedInterval r) noexcept
{
	assert(l.Overlaps(r));
	return { std::max(l.Lower(), r.Lower()), std::min(l.Upper(), r.Upper()) };
}

OpenInterval Hull(OpenInterval l, OpenInterval r) noexcept
{
	return { std::min(l.Lower(), r.Lower()), std::max(l.Upper(), r.Upper()) };
}

ClosedInterval Hull(ClosedInterval l, ClosedInterval r) noexcept
{
	return { std::min(l.Lower(), r.Lower()), std::max(l.Upper(), r.Upper()) };
}
