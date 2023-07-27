#pragma once
#include <algorithm>
#include <cstdint>
#include <string>

class OpenInterval
{
public:
	int lower, upper;

	OpenInterval(int lower, int upper) noexcept : lower(lower), upper(upper) {}

	bool operator==(const OpenInterval& o) const noexcept { return lower == o.lower && upper == o.upper; }
	bool operator!=(const OpenInterval& o) const noexcept { return lower != o.lower || upper != o.upper; }
	
	OpenInterval operator-() const noexcept { return { -upper, -lower }; }

	bool Contains(int value) const noexcept { return (lower < value) && (value < upper); }
};

template <typename T = int>
class ClosedInterval
{
public:
	T lower, upper;

	ClosedInterval(T lower, T upper) noexcept : lower(lower), upper(upper) {}

	bool operator==(const ClosedInterval& o) const noexcept { return lower == o.lower && upper == o.upper; }
	bool operator!=(const ClosedInterval& o) const noexcept { return lower != o.lower || upper != o.upper; }
		
	ClosedInterval operator-() const noexcept { return { -upper, -lower }; }

	bool Contains(T value) const noexcept { return (lower <= value) && (value <= upper); }
	bool Overlaps(const OpenInterval& o) const noexcept { return (upper > o.lower) && (lower < o.upper); }
};

template <typename T>
inline ClosedInterval<T> Intersection(const ClosedInterval<T>& l, const ClosedInterval<T>& r) noexcept { return { std::max(l.lower, r.lower), std::min(l.upper, r.upper) }; }

template <typename V> inline bool operator<(V value, const OpenInterval& i) noexcept { return value <= i.lower; }
template <typename V> inline bool operator>(V value, const OpenInterval& i) noexcept { return value >= i.upper; }
template <typename V> inline bool operator<(const OpenInterval& i, V value) noexcept { return i.upper <= value; }
template <typename V> inline bool operator>(const OpenInterval& i, V value) noexcept { return i.lower >= value; }
template <typename V, typename T> inline bool operator<(V value, const ClosedInterval<T>& i) noexcept { return value < i.lower; }
template <typename V, typename T> inline bool operator>(V value, const ClosedInterval<T>& i) noexcept { return value > i.upper; }
template <typename T, typename V> inline bool operator<(const ClosedInterval<T>& i, V value) noexcept { return i.upper < value; }
template <typename T, typename V> inline bool operator>(const ClosedInterval<T>& i, V value) noexcept { return i.lower > value; }
template <typename T> inline bool operator<(const OpenInterval& l, const ClosedInterval<T>& r) noexcept { return l.upper <= r.lower; }
template <typename T> inline bool operator>(const OpenInterval& l, const ClosedInterval<T>& r) noexcept { return l.lower >= r.upper; }
template <typename T> inline bool operator<(const ClosedInterval<T>& l, const OpenInterval& r) noexcept { return l.upper <= r.lower; }
template <typename T> inline bool operator>(const ClosedInterval<T>& l, const OpenInterval& r) noexcept { return l.lower >= r.upper; }
inline bool operator<(const OpenInterval& l, const OpenInterval& r) noexcept { return l.upper <= r.lower; }
inline bool operator>(const OpenInterval& l, const OpenInterval& r) noexcept { return l.lower >= r.upper; }
template <typename T> inline bool operator<(const ClosedInterval<T>& l, const ClosedInterval<T>& r) noexcept { return l.upper < r.lower; }
template <typename T> inline bool operator>(const ClosedInterval<T>& l, const ClosedInterval<T>& r) noexcept { return l.lower > r.upper; }

inline std::string to_string(const OpenInterval& i) { return "(" + std::to_string(i.lower) + "," + std::to_string(i.upper) + ")"; }
template <typename T> inline std::string to_string(const ClosedInterval<T>& i) { return "[" + std::to_string(i.lower) + "," + std::to_string(i.upper) + "]"; }
