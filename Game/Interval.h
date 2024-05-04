#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

class OpenInterval
{
public:
	Score lower, upper;

	OpenInterval() noexcept = default;
	OpenInterval(Score lower, Score upper) noexcept : lower(lower), upper(upper) {}

	bool operator==(const OpenInterval& o) const noexcept { return lower == o.lower && upper == o.upper; }
	bool operator!=(const OpenInterval& o) const noexcept { return lower != o.lower || upper != o.upper; }
	
	OpenInterval operator-() const noexcept { return { -upper, -lower }; }

	bool Contains(Score value) const noexcept { return (lower < value) && (value < upper); }
};

class ClosedInterval
{
public:
	Score lower, upper;

	ClosedInterval() noexcept = default;
	ClosedInterval(Score lower, Score upper) noexcept : lower(lower), upper(upper) {}

	bool operator==(const ClosedInterval& o) const noexcept { return lower == o.lower && upper == o.upper; }
	bool operator!=(const ClosedInterval& o) const noexcept { return lower != o.lower || upper != o.upper; }
		
	ClosedInterval operator-() const noexcept { return { -upper, -lower }; }

	bool Contains(Score value) const noexcept { return (lower <= value) && (value <= upper); }
	bool Overlaps(const OpenInterval& o) const noexcept { return (upper > o.lower) && (lower < o.upper); }
	bool Overlaps(const ClosedInterval& o) const noexcept { return (upper >= o.lower) && (lower <= o.upper); }
};

inline OpenInterval Intersection(const OpenInterval& l, const OpenInterval& r) noexcept
{
	assert(l.upper > r.lower);
	assert(l.lower < r.upper);
	return { std::max(l.lower, r.lower), std::min(l.upper, r.upper) };
}
inline ClosedInterval Intersection(const ClosedInterval& l, const ClosedInterval& r) noexcept
{
	assert(l.upper >= r.lower);
	assert(l.lower <= r.upper);
	return { std::max(l.lower, r.lower), std::min(l.upper, r.upper) };
}

inline bool operator<(auto value, const OpenInterval& i) noexcept { return value <= i.lower; }
inline bool operator>(auto value, const OpenInterval& i) noexcept { return value >= i.upper; }
inline bool operator<(const OpenInterval& i, auto value) noexcept { return i.upper <= value; }
inline bool operator>(const OpenInterval& i, auto value) noexcept { return i.lower >= value; }
inline bool operator<(auto value, const ClosedInterval& i) noexcept { return value < i.lower; }
inline bool operator>(auto value, const ClosedInterval& i) noexcept { return value > i.upper; }
inline bool operator<(const ClosedInterval& i, auto value) noexcept { return i.upper < value; }
inline bool operator>(const ClosedInterval& i, auto value) noexcept { return i.lower > value; }
inline bool operator<(const OpenInterval& l, const ClosedInterval& r) noexcept { return l.upper <= r.lower; }
inline bool operator>(const OpenInterval& l, const ClosedInterval& r) noexcept { return l.lower >= r.upper; }
inline bool operator<(const ClosedInterval& l, const OpenInterval& r) noexcept { return l.upper <= r.lower; }
inline bool operator>(const ClosedInterval& l, const OpenInterval& r) noexcept { return l.lower >= r.upper; }
inline bool operator<(const OpenInterval& l, const OpenInterval& r) noexcept { return l.upper <= r.lower; }
inline bool operator>(const OpenInterval& l, const OpenInterval& r) noexcept { return l.lower >= r.upper; }
inline bool operator<(const ClosedInterval& l, const ClosedInterval& r) noexcept { return l.upper < r.lower; }
inline bool operator>(const ClosedInterval& l, const ClosedInterval& r) noexcept { return l.lower > r.upper; }

inline std::string to_string(const OpenInterval& i) { return "(" + to_string(i.lower) + "," + to_string(i.upper) + ")"; }
inline std::string to_string(const ClosedInterval& i) { return "[" + to_string(i.lower) + "," + to_string(i.upper) + "]"; }
