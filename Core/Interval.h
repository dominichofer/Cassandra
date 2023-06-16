#pragma once
#include <algorithm>
#include <string>

class OpenInterval
{
public:
	int lower, upper;

	OpenInterval(int lower, int upper) noexcept : lower(lower), upper(upper) {}

	bool operator==(const OpenInterval&) const noexcept = default;
	bool operator!=(const OpenInterval&) const noexcept = default;
	
	OpenInterval operator-() const noexcept { return { -upper, -lower }; }

	bool Contains(int value) const noexcept { return (lower < value) and (value < upper); }
};

class ClosedInterval
{
public:
	int lower, upper;

	ClosedInterval(int lower, int upper) noexcept : lower(lower), upper(upper) {}

	bool operator==(const ClosedInterval&) const noexcept = default;
	bool operator!=(const ClosedInterval&) const noexcept = default;
		
	ClosedInterval operator-() const noexcept { return { -upper, -lower }; }

	bool Contains(int value) const noexcept { return (lower <= value) and (value <= upper); }
	bool Overlaps(const OpenInterval& o) const noexcept { return (upper > o.lower) and (lower < o.upper); }
};

inline bool operator<(int value, const OpenInterval& i) noexcept { return value <= i.lower; }
inline bool operator>(int value, const OpenInterval& i) noexcept { return value >= i.upper; }
inline bool operator<(const OpenInterval& i, int value) noexcept { return i.upper <= value; }
inline bool operator>(const OpenInterval& i, int value) noexcept { return i.lower >= value; }
inline bool operator<(int value, const ClosedInterval& i) noexcept { return value < i.lower; }
inline bool operator>(int value, const ClosedInterval& i) noexcept { return value > i.upper; }
inline bool operator<(const ClosedInterval& i, int value) noexcept { return i.upper < value; }
inline bool operator>(const ClosedInterval& i, int value) noexcept { return i.lower > value; }
inline bool operator<(const OpenInterval& l, const ClosedInterval& r) noexcept { return l.upper <= r.lower; }
inline bool operator>(const OpenInterval& l, const ClosedInterval& r) noexcept { return l.lower >= r.upper; }
inline bool operator<(const ClosedInterval& l, const OpenInterval& r) noexcept { return l.upper <= r.lower; }
inline bool operator>(const ClosedInterval& l, const OpenInterval& r) noexcept { return l.lower >= r.upper; }
inline bool operator<(const OpenInterval& l, const OpenInterval& r) noexcept { return l.upper <= r.lower; }
inline bool operator>(const OpenInterval& l, const OpenInterval& r) noexcept { return l.lower >= r.upper; }
inline bool operator<(const ClosedInterval& l, const ClosedInterval& r) noexcept { return l.upper < r.lower; }
inline bool operator>(const ClosedInterval& l, const ClosedInterval& r) noexcept { return l.lower > r.upper; }


inline std::string to_string(const OpenInterval& i) { return "(" + std::to_string(i.lower) + "," + std::to_string(i.upper) + ")"; }
inline std::string to_string(const ClosedInterval& i) { return "[" + std::to_string(i.lower) + "," + std::to_string(i.upper) + "]"; }
