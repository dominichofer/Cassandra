#pragma once
#include "Core/Core.h"
#include <algorithm>
#include <cassert>
#include <string>

class Interval
{
protected:
	int m_lower, m_upper;
public:
	Interval(int lower, int upper) noexcept : m_lower(lower), m_upper(upper) {}

	[[nodiscard]] bool operator==(const Interval&) const noexcept = default;
	[[nodiscard]] bool operator!=(const Interval&) const noexcept = default;

	[[nodiscard]] int lower() const noexcept { return m_lower; }
	[[nodiscard]] int upper() const noexcept { return m_upper; }

	[[nodiscard]] int clamp(int s) const noexcept { return std::clamp(s, m_lower, m_upper); }

	[[nodiscard]] virtual bool Constrained() const noexcept = 0;

	bool TryIncreaseLower(int s) { if (s > m_lower) { m_lower = s; assert(Constrained()); return true; } return false; }
	bool TryDecreaseLower(int s) { if (s < m_lower) { m_lower = s; assert(Constrained()); return true; } return false; }
	bool TryIncreaseUpper(int s) { if (s > m_upper) { m_upper = s; assert(Constrained()); return true; } return false; }
	bool TryDecreaseUpper(int s) { if (s < m_upper) { m_upper = s; assert(Constrained()); return true; } return false; }
};

class ClosedInterval;

class OpenInterval final : public Interval
{
	[[nodiscard]] bool Constrained() const noexcept override { return (-inf_score <= m_lower) && (m_lower < m_upper) && (m_upper <= +inf_score); }

public:
	OpenInterval() = delete;
	OpenInterval(int lower, int upper) noexcept : Interval(lower, upper) { assert(Constrained()); }
	explicit OpenInterval(ClosedInterval) noexcept;

	[[nodiscard]] static OpenInterval Whole() { return { min_score, max_score }; }
	
	[[nodiscard]] OpenInterval operator-() const noexcept { return { -m_upper, -m_lower }; }
	OpenInterval& operator-=(ClosedInterval) noexcept;

	[[nodiscard]] bool empty() const noexcept { return m_lower + 1 == m_upper; }
	[[nodiscard]] int Span() const noexcept { return m_upper - m_lower; }

	[[nodiscard]] bool Contains(int s) const noexcept { return (m_lower < s) && (s < m_upper); }
	[[nodiscard]] bool Contains(OpenInterval o) const noexcept { return (m_lower <= o.m_lower) && (o.m_upper <= m_upper); }
	[[nodiscard]] bool Contains(ClosedInterval) const noexcept;

	[[nodiscard]] bool Overlaps(OpenInterval o) const noexcept { return (m_lower + 1 < o.upper()) && (m_upper - 1 > o.lower()); }
	[[nodiscard]] bool Overlaps(ClosedInterval) const noexcept;
};

class ClosedInterval final : public Interval
{
	[[nodiscard]] bool Constrained() const noexcept override { return (min_score <= m_lower) && (m_lower <= m_upper) && (m_upper <= max_score); }

public:
	ClosedInterval() = delete;
	ClosedInterval(int lower, int upper) noexcept : Interval(lower, upper) { assert(Constrained()); }
	// explicit ClosedInterval(OpenInterval o) noexcept : ClosedInterval(o.m_lower + 1, o.m_upper - 1) {}

	[[nodiscard]] static ClosedInterval Whole() noexcept { return { min_score, max_score }; }
		
	[[nodiscard]] ClosedInterval operator-() const noexcept { return { -m_upper, -m_lower }; }
	ClosedInterval& operator-=(OpenInterval) noexcept;

	[[nodiscard]] bool IsSingleton() const noexcept { return m_lower == m_upper; }
	[[nodiscard]] int Span() const noexcept { return m_upper - m_lower; }
	
	[[nodiscard]] bool Contains(int s) const noexcept { return (m_lower <= s) && (s <= m_upper); }
	[[nodiscard]] bool Contains(OpenInterval o) const noexcept { return OpenInterval(*this).Contains(o); }
	[[nodiscard]] bool Contains(ClosedInterval o) const noexcept { return (m_lower <= o.m_lower) && (o.m_upper <= m_upper); }

	[[nodiscard]] bool Overlaps(OpenInterval o) const noexcept { return o.Overlaps(*this); }
	[[nodiscard]] bool Overlaps(ClosedInterval o) const noexcept { return (m_lower <= o.upper()) && (m_upper >= o.lower()); }
};

[[nodiscard]] inline bool operator<(OpenInterval l, int r) noexcept { return l.upper() <= r; }
[[nodiscard]] inline bool operator<(int l, OpenInterval r) noexcept { return l <= r.lower(); }
[[nodiscard]] inline bool operator<(ClosedInterval l, int r) noexcept { return l.upper() < r; }
[[nodiscard]] inline bool operator<(int l, ClosedInterval r) noexcept { return l < r.lower(); }
[[nodiscard]] inline bool operator<(OpenInterval l, OpenInterval r) noexcept { return l.upper() - 1 <= r.lower(); }
[[nodiscard]] inline bool operator<(OpenInterval l, ClosedInterval r) noexcept { return l.upper() <= r.lower(); }
[[nodiscard]] inline bool operator<(ClosedInterval l, OpenInterval r) noexcept { return l.upper() <= r.lower(); }
[[nodiscard]] inline bool operator<(ClosedInterval l, ClosedInterval r) noexcept { return l.upper() < r.lower(); }

[[nodiscard]] inline bool operator>(OpenInterval l, int r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(int l, OpenInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(ClosedInterval l, int r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(int l, ClosedInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(OpenInterval l, OpenInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(OpenInterval l, ClosedInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(ClosedInterval l, OpenInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(ClosedInterval l, ClosedInterval r) noexcept { return r < l; }

[[nodiscard]] inline OpenInterval operator-(OpenInterval l, ClosedInterval r) noexcept { return l -= r; }
[[nodiscard]] inline ClosedInterval operator-(ClosedInterval l, OpenInterval r) noexcept { return l -= r; }


[[nodiscard]] OpenInterval Overlap(OpenInterval l, OpenInterval r) noexcept;
[[nodiscard]] ClosedInterval Overlap(ClosedInterval l, ClosedInterval r) noexcept;

[[nodiscard]] OpenInterval Hull(OpenInterval l, OpenInterval r) noexcept;
[[nodiscard]] ClosedInterval Hull(ClosedInterval l, ClosedInterval r) noexcept;

[[nodiscard]] inline int Span(OpenInterval i) noexcept { return i.Span(); }
[[nodiscard]] inline int Span(ClosedInterval i) noexcept { return i.Span(); }


[[nodiscard]] inline std::string to_string(const OpenInterval& i) { return "(" + std::to_string(i.lower()) + "," + std::to_string(i.upper()) + ")"; }
[[nodiscard]] inline std::string to_string(const ClosedInterval& i) { return "[" + std::to_string(i.lower()) + "," + std::to_string(i.upper()) + "]"; }