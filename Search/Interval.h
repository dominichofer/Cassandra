#pragma once
#include <algorithm>
#include <string>

class Interval
{
protected:
	int lower, upper;
public:
	Interval(int lower, int upper) noexcept : lower(lower), upper(upper) {}

	bool operator==(const Interval&) const noexcept = default;
	bool operator!=(const Interval&) const noexcept = default;

	int Lower() const noexcept { return lower; }
	int Upper() const noexcept { return upper; }

	int clamp(int value) const { return std::clamp(value, lower, upper); }

	void TryIncreaseLower(int value) { if (value > lower) lower = value; }
	void TryDecreaseLower(int value) { if (value < lower) lower = value; }
	void TryIncreaseUpper(int value) { if (value > upper) upper = value; }
	void TryDecreaseUpper(int value) { if (value < upper) upper = value; }
};

class ClosedInterval;

class OpenInterval final : public Interval
{
public:
	OpenInterval() = delete;
	OpenInterval(int lower, int upper) noexcept : Interval(lower, upper) {}
	explicit OpenInterval(ClosedInterval o) noexcept;
	
	OpenInterval operator-() const noexcept { return { -upper, -lower }; }
	OpenInterval& operator-=(ClosedInterval) noexcept;

	bool empty() const noexcept { return lower + 1 == upper; }

	bool Contains(int value) const noexcept { return (lower < value) and (value < upper); }
	bool Contains(OpenInterval o) const noexcept { return (lower <= o.lower) and (o.upper <= upper); }
	bool Contains(ClosedInterval) const noexcept;

	bool Overlaps(OpenInterval o) const noexcept { return (lower + 1 < o.Upper()) and (upper - 1 > o.Lower()); }
	bool Overlaps(ClosedInterval) const noexcept;
};

class ClosedInterval final : public Interval
{
public:
	ClosedInterval() = delete;
	ClosedInterval(int lower, int upper) noexcept : Interval(lower, upper) {}
	ClosedInterval(int value) noexcept : Interval(value, value) {}
	explicit ClosedInterval(OpenInterval o) noexcept : ClosedInterval(o.Lower() + 1, o.Upper() - 1) {}
		
	ClosedInterval operator-() const noexcept { return { -upper, -lower }; }
	ClosedInterval& operator-=(OpenInterval) noexcept;

	bool IsSingleton() const noexcept { return lower == upper; }
	
	bool Contains(int s) const noexcept { return (lower <= s) and (s <= upper); }
	bool Contains(OpenInterval o) const noexcept { return OpenInterval(*this).Contains(o); }
	bool Contains(ClosedInterval o) const noexcept { return (lower <= o.lower) and (o.upper <= upper); }

	bool Overlaps(OpenInterval o) const noexcept { return o.Overlaps(*this); }
	bool Overlaps(ClosedInterval o) const noexcept { return (lower <= o.Upper()) and (upper >= o.Lower()); }
};

inline bool operator<(OpenInterval l, int r) noexcept { return l.Upper() <= r; }
inline bool operator<(int l, OpenInterval r) noexcept { return l <= r.Lower(); }
inline bool operator<(ClosedInterval l, int r) noexcept { return l.Upper() < r; }
inline bool operator<(int l, ClosedInterval r) noexcept { return l < r.Lower(); }
inline bool operator<(OpenInterval l, OpenInterval r) noexcept { return l.Upper() - 1 <= r.Lower(); }
inline bool operator<(OpenInterval l, ClosedInterval r) noexcept { return l.Upper() <= r.Lower(); }
inline bool operator<(ClosedInterval l, OpenInterval r) noexcept { return l.Upper() <= r.Lower(); }
inline bool operator<(ClosedInterval l, ClosedInterval r) noexcept { return l.Upper() < r.Lower(); }

inline bool operator>(OpenInterval l, int r) noexcept { return r < l; }
inline bool operator>(int l, OpenInterval r) noexcept { return r < l; }
inline bool operator>(ClosedInterval l, int r) noexcept { return r < l; }
inline bool operator>(int l, ClosedInterval r) noexcept { return r < l; }
inline bool operator>(OpenInterval l, OpenInterval r) noexcept { return r < l; }
inline bool operator>(OpenInterval l, ClosedInterval r) noexcept { return r < l; }
inline bool operator>(ClosedInterval l, OpenInterval r) noexcept { return r < l; }
inline bool operator>(ClosedInterval l, ClosedInterval r) noexcept { return r < l; }

inline OpenInterval operator-(OpenInterval l, ClosedInterval r) noexcept { return l -= r; }
inline ClosedInterval operator-(ClosedInterval l, OpenInterval r) noexcept { return l -= r; }


OpenInterval Overlap(OpenInterval l, OpenInterval r) noexcept;
ClosedInterval Overlap(ClosedInterval l, ClosedInterval r) noexcept;

OpenInterval Hull(OpenInterval l, OpenInterval r) noexcept;
ClosedInterval Hull(ClosedInterval l, ClosedInterval r) noexcept;


inline std::string to_string(const OpenInterval& i) { return "(" + std::to_string(i.Lower()) + "," + std::to_string(i.Upper()) + ")"; }
inline std::string to_string(const ClosedInterval& i) { return "[" + std::to_string(i.Lower()) + "," + std::to_string(i.Upper()) + "]"; }
