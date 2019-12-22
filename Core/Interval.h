#pragma once
#include <algorithm>

using Score = int;

class InclusiveInterval
{
	bool Constraint() const noexcept { return lower <= upper; }
public:
	// TODO: Because members are public, the constraint can be violated.
	Score lower, upper;

	InclusiveInterval() = delete;
	InclusiveInterval(Score lower, Score upper) noexcept : lower(lower), upper(upper) {}

	static const InclusiveInterval Full;

	[[nodiscard]] bool operator==(InclusiveInterval o) const noexcept { return (upper == o.upper) && (lower == o.lower); }
	[[nodiscard]] bool operator!=(InclusiveInterval o) const noexcept { return (upper != o.upper) || (lower != o.lower); }
	[[nodiscard]] bool operator<(InclusiveInterval o) const noexcept { return upper < o.lower; }
	[[nodiscard]] bool operator>(InclusiveInterval o) const noexcept { return lower > o.upper; }

	[[nodiscard]] bool Contains(Score s) const noexcept { return (lower <= s) && (s <= upper); }

	// Inverted interval.
	[[nodiscard]] InclusiveInterval operator-() const noexcept { return { -upper, -lower }; }
};

[[nodiscard]] inline InclusiveInterval Intersection(InclusiveInterval l, InclusiveInterval r) noexcept
{
	return { std::max(l.lower, r.lower), std::min(l.upper, r.upper) };
}

[[nodiscard]] inline bool operator<(InclusiveInterval r, Score s) noexcept { return r.upper < s; }
[[nodiscard]] inline bool operator>(InclusiveInterval r, Score s) noexcept { return r.lower > s; }
[[nodiscard]] inline bool operator<(Score s, InclusiveInterval r) noexcept { return s < r.lower; }
[[nodiscard]] inline bool operator>(Score s, InclusiveInterval r) noexcept { return s > r.upper; }

class ExclusiveInterval
{
	bool Constraint() const noexcept { return lower < upper; }
public:
	// TODO: Because members are public, the constraint can be violated.
	Score lower, upper;

	ExclusiveInterval() = delete;
	ExclusiveInterval(Score lower, Score upper) noexcept : lower(lower), upper(upper) {}

	static const ExclusiveInterval Full;

	[[nodiscard]] bool operator==(ExclusiveInterval o) const noexcept { return (upper == o.upper) && (lower == o.lower); }
	[[nodiscard]] bool operator!=(ExclusiveInterval o) const noexcept { return (upper != o.upper) || (lower != o.lower); }
	[[nodiscard]] bool operator<(ExclusiveInterval o) const noexcept { return upper <= o.lower; }
	[[nodiscard]] bool operator>(ExclusiveInterval o) const noexcept { return lower >= o.upper; }

	[[nodiscard]] bool Contains(Score s) const noexcept { return (lower < s) && (s < upper); }

	// Inverted interval.
	[[nodiscard]] ExclusiveInterval operator-() const noexcept { return { -upper, -lower }; }
};

[[nodiscard]] inline ExclusiveInterval Intersection(ExclusiveInterval l, ExclusiveInterval r) noexcept
{
	return { std::max(l.lower, r.lower), std::min(l.upper, r.upper) };
}

[[nodiscard]] inline bool operator<(ExclusiveInterval r, Score s) noexcept { return r.upper <= s; }
[[nodiscard]] inline bool operator>(ExclusiveInterval r, Score s) noexcept { return r.lower >= s; }
[[nodiscard]] inline bool operator<(Score s, ExclusiveInterval r) noexcept { return s <= r.lower; }
[[nodiscard]] inline bool operator>(Score s, ExclusiveInterval r) noexcept { return s >= r.upper; }

[[nodiscard]] inline bool operator<(ExclusiveInterval e, InclusiveInterval i) noexcept { return e.upper <= i.lower; }
[[nodiscard]] inline bool operator>(ExclusiveInterval e, InclusiveInterval i) noexcept { return e.lower >= i.upper; }
[[nodiscard]] inline bool operator<(InclusiveInterval i, ExclusiveInterval e) noexcept { return i.upper <= e.lower; }
[[nodiscard]] inline bool operator>(InclusiveInterval i, ExclusiveInterval e) noexcept { return i.lower >= e.upper; }