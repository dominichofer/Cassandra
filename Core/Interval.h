#pragma once
#include <algorithm>
#include <cassert>

class Score
{
	int value{};
public:
	static const Score Min;
	static const Score Max;
	static const Score Infinity;

	Score() = default;
	constexpr Score(int value) noexcept : value(value) {}

	[[nodiscard]] constexpr operator int() const noexcept { return value; }

	[[nodiscard]] Score operator++() noexcept { return ++value; }
	[[nodiscard]] Score operator++(int) noexcept { return value++; }
	[[nodiscard]] Score operator--() noexcept { return --value; }
	[[nodiscard]] Score operator--(int) noexcept { return value--; }

	[[nodiscard]] friend constexpr bool operator==(Score l, Score r) noexcept { return l.value == r.value; }
	[[nodiscard]] friend constexpr bool operator==(int   l, Score r) noexcept { return l == r.value; }
	[[nodiscard]] friend constexpr bool operator==(Score l, int   r) noexcept { return l.value == r; }
	[[nodiscard]] friend constexpr bool operator!=(Score l, Score r) noexcept { return l.value != r.value; }
	[[nodiscard]] friend constexpr bool operator!=(int   l, Score r) noexcept { return l != r.value; }
	[[nodiscard]] friend constexpr bool operator!=(Score l, int   r) noexcept { return l.value != r; }
	[[nodiscard]] friend constexpr bool operator<(Score l, Score r) noexcept { return l.value < r.value; }
	[[nodiscard]] friend constexpr bool operator<(int   l, Score r) noexcept { return l < r.value; }
	[[nodiscard]] friend constexpr bool operator<(Score l, int   r) noexcept { return l.value < r; }
	[[nodiscard]] friend constexpr bool operator>(Score l, Score r) noexcept { return l.value > r.value; }
	[[nodiscard]] friend constexpr bool operator>(int   l, Score r) noexcept { return l > r.value; }
	[[nodiscard]] friend constexpr bool operator>(Score l, int   r) noexcept { return l.value > r; }
	[[nodiscard]] friend constexpr bool operator<=(Score l, Score r) noexcept { return l.value <= r.value; }
	[[nodiscard]] friend constexpr bool operator<=(int   l, Score r) noexcept { return l <= r.value; }
	[[nodiscard]] friend constexpr bool operator<=(Score l, int   r) noexcept { return l.value <= r; }
	[[nodiscard]] friend constexpr bool operator>=(Score l, Score r) noexcept { return l.value >= r.value; }
	[[nodiscard]] friend constexpr bool operator>=(int   l, Score r) noexcept { return l >= r.value; }
	[[nodiscard]] friend constexpr bool operator>=(Score l, int   r) noexcept { return l.value >= r; }

	[[nodiscard]] friend constexpr Score operator+(Score l, Score r) noexcept { return l.value + r.value; }
	[[nodiscard]] friend constexpr Score operator+(int   l, Score r) noexcept { return l + r.value; }
	[[nodiscard]] friend constexpr Score operator+(Score l, int   r) noexcept { return l.value + r; }
	[[nodiscard]] friend constexpr Score operator-(Score l, Score r) noexcept { return l.value - r.value; }
	[[nodiscard]] friend constexpr Score operator-(int   l, Score r) noexcept { return l - r.value; }
	[[nodiscard]] friend constexpr Score operator-(Score l, int   r) noexcept { return l.value - r; }
	//[[nodiscard]] friend constexpr Score operator*(Score l, Score r) noexcept { return l.value * r.value; }
	//[[nodiscard]] friend constexpr Score operator*(int   l, Score r) noexcept { return l * r.value; }
	//[[nodiscard]] friend constexpr Score operator*(Score l, int   r) noexcept { return l.value * r; }
	//[[nodiscard]] friend constexpr Score operator/(Score l, Score r) noexcept { return l.value / r.value; }
	//[[nodiscard]] friend constexpr Score operator/(int   l, Score r) noexcept { return l / r.value; }
	//[[nodiscard]] friend constexpr Score operator/(Score l, int   r) noexcept { return l.value / r; }
};

[[nodiscard]] constexpr Score operator-(Score o) noexcept { return -static_cast<int>(o); }
[[nodiscard]] constexpr Score operator+(Score o) noexcept { return +static_cast<int>(o); }

class ClosedInterval
{
	[[nodiscard]] bool Constraint() const noexcept { return (-Score::Infinity < lower) && (lower <= upper) && (upper < +Score::Infinity); }
public:
	// TODO: Because members are public, the constraint can be violated.
	Score lower, upper;

	ClosedInterval() = delete;
	ClosedInterval(Score lower, Score upper) noexcept : lower(lower), upper(upper) { assert(Constraint()); }

	static const ClosedInterval Full;

	[[nodiscard]] bool operator==(ClosedInterval o) const noexcept { return (upper == o.upper) && (lower == o.lower); }
	[[nodiscard]] bool operator!=(ClosedInterval o) const noexcept { return (upper != o.upper) || (lower != o.lower); }
	[[nodiscard]] bool operator<(ClosedInterval o) const noexcept { return upper < o.lower; }
	[[nodiscard]] bool operator>(ClosedInterval o) const noexcept { return lower > o.upper; }

	[[nodiscard]] bool Contains(Score s) const noexcept { return (lower <= s) && (s <= upper); }

	// Inverted interval.
	[[nodiscard]] ClosedInterval operator-() const noexcept { return { -upper, -lower }; }
};

[[nodiscard]] inline ClosedInterval Intersection(ClosedInterval l, ClosedInterval r) noexcept
{
	return { std::max(l.lower, r.lower), std::min(l.upper, r.upper) };
}

[[nodiscard]] inline bool operator<(ClosedInterval r, Score s) noexcept { return r.upper < s; }
[[nodiscard]] inline bool operator>(ClosedInterval r, Score s) noexcept { return r.lower > s; }
[[nodiscard]] inline bool operator<(Score s, ClosedInterval r) noexcept { return s < r.lower; }
[[nodiscard]] inline bool operator>(Score s, ClosedInterval r) noexcept { return s > r.upper; }

class OpenInterval
{
	[[nodiscard]] bool Constraint() const noexcept { return (-Score::Infinity <= lower) && (lower < upper) && (upper <= +Score::Infinity); }
public:
	// TODO: Because members are public, the constraint can be violated.
	Score lower, upper;

	OpenInterval() = delete;
	OpenInterval(Score lower, Score upper) noexcept : lower(lower), upper(upper) {} // TODO: Add assert Constrint()!

	static const OpenInterval Full;

	[[nodiscard]] bool operator==(OpenInterval o) const noexcept { return (upper == o.upper) && (lower == o.lower); }
	[[nodiscard]] bool operator!=(OpenInterval o) const noexcept { return (upper != o.upper) || (lower != o.lower); }
	[[nodiscard]] bool operator<(OpenInterval o) const noexcept { return upper <= o.lower; }
	[[nodiscard]] bool operator>(OpenInterval o) const noexcept { return lower >= o.upper; }

	[[nodiscard]] bool Contains(Score s) const noexcept { return (lower < s) && (s < upper); }

	// Inverted interval.
	[[nodiscard]] OpenInterval operator-() const noexcept { return { -upper, -lower }; }
};

[[nodiscard]] inline OpenInterval Intersection(OpenInterval l, OpenInterval r) noexcept
{
	return { std::max(l.lower, r.lower), std::min(l.upper, r.upper) };
}

[[nodiscard]] inline bool operator<(OpenInterval r, Score s) noexcept { return r.upper <= s; }
[[nodiscard]] inline bool operator>(OpenInterval r, Score s) noexcept { return r.lower >= s; }
[[nodiscard]] inline bool operator<(Score s, OpenInterval r) noexcept { return s <= r.lower; }
[[nodiscard]] inline bool operator>(Score s, OpenInterval r) noexcept { return s >= r.upper; }

[[nodiscard]] inline bool operator<(OpenInterval e, ClosedInterval i) noexcept { return e.upper <= i.lower; }
[[nodiscard]] inline bool operator>(OpenInterval e, ClosedInterval i) noexcept { return e.lower >= i.upper; }
[[nodiscard]] inline bool operator<(ClosedInterval i, OpenInterval e) noexcept { return i.upper <= e.lower; }
[[nodiscard]] inline bool operator>(ClosedInterval i, OpenInterval e) noexcept { return i.lower >= e.upper; }