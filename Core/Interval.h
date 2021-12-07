#pragma once
#include "Position.h"
#include <algorithm>
#include <cassert>
#include <string>

class Interval
{
protected:
	int m_lower, m_upper;
public:
	Interval(int lower, int upper) noexcept : m_lower(lower), m_upper(upper) {}

	bool operator==(const Interval&) const noexcept = default;
	bool operator!=(const Interval&) const noexcept = default;

	int lower() const noexcept { return m_lower; }
	int upper() const noexcept { return m_upper; }

	int clamp(int s) const noexcept { return std::clamp(s, m_lower, m_upper); }

	virtual bool Constrained() const noexcept = 0;

	void TryIncreaseLower(int s) { if (s > m_lower) m_lower = s; }
	void TryDecreaseLower(int s) { if (s < m_lower) m_lower = s; }
	void TryIncreaseUpper(int s) { if (s > m_upper) m_upper = s; }
	void TryDecreaseUpper(int s) { if (s < m_upper) m_upper = s; }
};

class ClosedInterval;

class OpenInterval final : public Interval
{
	bool Constrained() const noexcept override { return (-inf_score <= m_lower) && (m_lower < m_upper) && (m_upper <= +inf_score); }

public:
	OpenInterval() = delete;
	OpenInterval(int lower, int upper) noexcept : Interval(lower, upper) { assert(Constrained()); }
	explicit OpenInterval(ClosedInterval) noexcept;

	static OpenInterval Whole() { return { min_score, max_score }; }
	
	OpenInterval operator-() const noexcept { return { -m_upper, -m_lower }; }
	OpenInterval& operator-=(ClosedInterval) noexcept;

	bool empty() const noexcept { return m_lower + 1 == m_upper; }

	bool Contains(int s) const noexcept { return (m_lower < s) && (s < m_upper); }
	bool Contains(OpenInterval o) const noexcept { return (m_lower <= o.m_lower) && (o.m_upper <= m_upper); }
	bool Contains(ClosedInterval) const noexcept;

	bool Overlaps(OpenInterval o) const noexcept { return (m_lower + 1 < o.upper()) && (m_upper - 1 > o.lower()); }
	bool Overlaps(ClosedInterval) const noexcept;
};

class ClosedInterval final : public Interval
{
	bool Constrained() const noexcept override { return (min_score <= m_lower) && (m_lower <= m_upper) && (m_upper <= max_score); }

public:
	ClosedInterval() = delete;
	ClosedInterval(int lower, int upper) noexcept : Interval(lower, upper) { assert(Constrained()); }
	// explicit ClosedInterval(OpenInterval o) noexcept : ClosedInterval(o.m_lower + 1, o.m_upper - 1) {}

	static ClosedInterval Whole() noexcept { return { min_score, max_score }; }
		
	ClosedInterval operator-() const noexcept { return { -m_upper, -m_lower }; }
	ClosedInterval& operator-=(OpenInterval) noexcept;

	bool IsSingleton() const noexcept { return m_lower == m_upper; }
	
	bool Contains(int s) const noexcept { return (m_lower <= s) && (s <= m_upper); }
	bool Contains(OpenInterval o) const noexcept { return OpenInterval(*this).Contains(o); }
	bool Contains(ClosedInterval o) const noexcept { return (m_lower <= o.m_lower) && (o.m_upper <= m_upper); }

	bool Overlaps(OpenInterval o) const noexcept { return o.Overlaps(*this); }
	bool Overlaps(ClosedInterval o) const noexcept { return (m_lower <= o.upper()) && (m_upper >= o.lower()); }
};

inline bool operator<(OpenInterval l, int r) noexcept { return l.upper() <= r; }
inline bool operator<(int l, OpenInterval r) noexcept { return l <= r.lower(); }
inline bool operator<(ClosedInterval l, int r) noexcept { return l.upper() < r; }
inline bool operator<(int l, ClosedInterval r) noexcept { return l < r.lower(); }
inline bool operator<(OpenInterval l, OpenInterval r) noexcept { return l.upper() - 1 <= r.lower(); }
inline bool operator<(OpenInterval l, ClosedInterval r) noexcept { return l.upper() <= r.lower(); }
inline bool operator<(ClosedInterval l, OpenInterval r) noexcept { return l.upper() <= r.lower(); }
inline bool operator<(ClosedInterval l, ClosedInterval r) noexcept { return l.upper() < r.lower(); }

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


inline std::string to_string(const OpenInterval& i) { return "(" + std::to_string(i.lower()) + "," + std::to_string(i.upper()) + ")"; }
inline std::string to_string(const ClosedInterval& i) { return "[" + std::to_string(i.lower()) + "," + std::to_string(i.upper()) + "]"; }