#pragma once
#include "Core/Core.h"
#include <algorithm>
#include <cassert>
#include <compare>

struct Interval
{
	Score m_lower, m_upper;

	Interval(Score lower, Score upper) noexcept : m_lower(lower), m_upper(upper) {}

	[[nodiscard]] Score lower() const { return m_lower; }
	[[nodiscard]] Score upper() const { return m_upper; }
	
	[[nodiscard]] bool operator==(const Interval& o) const noexcept = default;
	[[nodiscard]] bool operator!=(const Interval & o) const noexcept = default;

	[[nodiscard]] Score clamp(Score s) const { return std::clamp(s, m_lower, m_upper); }

	[[nodiscard]] virtual bool Constrained() const noexcept = 0;

	bool try_increase_lower(Score s) { if (s > m_lower) { m_lower = s; assert(Constrained()); return true; } return false; }
	bool try_decrease_lower(Score s) { if (s < m_lower) { m_lower = s; assert(Constrained()); return true; } return false; }
	bool try_increase_upper(Score s) { if (s > m_upper) { m_upper = s; assert(Constrained()); return true; } return false; }
	bool try_decrease_upper(Score s) { if (s < m_upper) { m_upper = s; assert(Constrained()); return true; } return false; }
};

class ClosedInterval;

class OpenInterval final : public Interval
{
	[[nodiscard]] bool Constrained() const noexcept override { return (-infinity <= m_lower) && (m_lower < m_upper) && (m_upper <= +infinity); }

public:
	OpenInterval() = delete;
	OpenInterval(Score lower, Score upper) noexcept : Interval(lower, upper) { assert(Constrained()); }
	explicit OpenInterval(ClosedInterval) noexcept;

	[[nodiscard]] static OpenInterval Whole() { return { -infinity, +infinity }; }
	
	[[nodiscard]] OpenInterval operator-() const noexcept { return { -m_upper, -m_lower }; }
	OpenInterval& operator-=(ClosedInterval) noexcept;

	[[nodiscard]] bool empty() const noexcept { return m_lower + 1 == m_upper; }

	[[nodiscard]] bool Contains(Score s) const noexcept { return (m_lower < s) && (s < m_upper); }
	[[nodiscard]] bool Contains(OpenInterval o) const noexcept { return (m_lower <= o.m_lower) && (o.m_upper <= m_upper); }
	[[nodiscard]] bool Contains(ClosedInterval) const noexcept;

	[[nodiscard]] bool Overlaps(OpenInterval) const noexcept;
	[[nodiscard]] bool Overlaps(ClosedInterval) const noexcept;
};

class ClosedInterval final : public Interval
{
	// TODO: Does it need to include infinity?
	[[nodiscard]] bool Constrained() const noexcept override { return (min_score <= m_lower) && (m_lower <= m_upper) && (m_upper <= max_score); }

public:
	ClosedInterval() = delete;
	ClosedInterval(Score lower, Score upper) noexcept : Interval(lower, upper) { assert(Constrained()); }
	// explicit ClosedInterval(OpenInterval o) noexcept : ClosedInterval(o.m_lower + 1, o.m_upper - 1) {}

	[[nodiscard]] static ClosedInterval Whole() noexcept { return { min_score, max_score }; }
		
	[[nodiscard]] ClosedInterval operator-() const noexcept { return { -m_upper, -m_lower }; }
	ClosedInterval& operator-=(OpenInterval) noexcept;

	[[nodiscard]] bool IsSingleton() const noexcept { return m_lower == m_upper; }
	
	[[nodiscard]] bool Contains(Score s) const noexcept { return (m_lower <= s) && (s <= m_upper); }
	[[nodiscard]] bool Contains(OpenInterval o) const noexcept { return OpenInterval(*this).Contains(o); }
	[[nodiscard]] bool Contains(ClosedInterval o) const noexcept { return (m_lower <= o.m_lower) && (o.m_upper <= m_upper); }

	[[nodiscard]] bool Overlaps(OpenInterval o) const noexcept;
	[[nodiscard]] bool Overlaps(ClosedInterval o) const noexcept;
};

[[nodiscard]] inline bool operator<(OpenInterval l, Score r) noexcept { return l.upper() <= r; }
[[nodiscard]] inline bool operator<(Score l, OpenInterval r) noexcept { return l <= r.lower(); }
[[nodiscard]] inline bool operator<(ClosedInterval l, Score r) noexcept { return l.upper() < r; }
[[nodiscard]] inline bool operator<(Score l, ClosedInterval r) noexcept { return l < r.lower(); }
[[nodiscard]] inline bool operator<(OpenInterval l, OpenInterval r) noexcept { return l.upper() - 1 <= r.lower(); }
[[nodiscard]] inline bool operator<(OpenInterval l, ClosedInterval r) noexcept { return l.upper() <= r.lower(); }
[[nodiscard]] inline bool operator<(ClosedInterval l, OpenInterval r) noexcept { return l.upper() <= r.lower(); }
[[nodiscard]] inline bool operator<(ClosedInterval l, ClosedInterval r) noexcept { return l.upper() < r.lower(); }

[[nodiscard]] inline bool operator>(OpenInterval l, Score r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(Score l, OpenInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(ClosedInterval l, Score r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(Score l, ClosedInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(OpenInterval l, OpenInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(OpenInterval l, ClosedInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(ClosedInterval l, OpenInterval r) noexcept { return r < l; }
[[nodiscard]] inline bool operator>(ClosedInterval l, ClosedInterval r) noexcept { return r < l; }

[[nodiscard]] inline OpenInterval operator-(OpenInterval l, ClosedInterval r) noexcept { l -= r; return l; }
[[nodiscard]] inline ClosedInterval operator-(ClosedInterval l, OpenInterval r) noexcept { l -= r; return l; }


[[nodiscard]] OpenInterval Overlap(OpenInterval l, OpenInterval r) noexcept;
[[nodiscard]] ClosedInterval Overlap(ClosedInterval l, ClosedInterval r) noexcept;

[[nodiscard]] OpenInterval Hull(OpenInterval l, OpenInterval r) noexcept;
[[nodiscard]] ClosedInterval Hull(ClosedInterval l, ClosedInterval r) noexcept;
