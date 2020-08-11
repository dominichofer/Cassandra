#pragma once
#include "Core/Core.h"
#include "Interval.h"
#include <cassert>
#include <cstdint>
#include <cmath>

namespace Search
{	
	class Selectivity
	{
		//  Phi       z
		// 0.00 ~=  0.00%
		// 0.10 ~=  7.97%
		// 0.20 ~= 15.85%
		// 0.30 ~= 23.58%
		// 0.40 ~= 31.08%
		// 0.50 ~= 38.29%
		// 0.60 ~= 45.15%
		// 0.70 ~= 51.61%
		// 0.80 ~= 57.63%
		// 0.90 ~= 63.19%
		// 1.00 ~= 68.27%
		// 1.10 ~= 72.87% <- edax
		// 1.20 ~= 76.99%
		// 1.30 ~= 80.64%
		// 1.40 ~= 83.85%
		// 1.50 ~= 86.64% <- edax, logistello
		// 1.60 ~= 89.04%
		// 1.70 ~= 91.09%
		// 1.80 ~= 92.81%
		// 1.90 ~= 94.26%
		// 2.00 ~= 95.45% <- edax
		// 2.10 ~= 96.43%
		// 2.20 ~= 97.22%
		// 2.30 ~= 97.86%
		// 2.40 ~= 98.36%
		// 2.50 ~= 98.76%
		// 2.60 ~= 99.07% <- edax
		// 2.70 ~= 99.31%
		// 2.80 ~= 99.49%
		// 2.90 ~= 99.63%
		// 3.00 ~= 99.73%
		// 3.10 ~= 99.81%
		// 3.20 ~= 99.86%
		// 3.30 ~= 99.90% <- edax

		double Phi(double z) { return std::erf(z / std::sqrt(2)); } // TODO: Why is this private and why is it there? it's unused.
	public:
		float quantile;

		Selectivity() = delete;
		constexpr explicit Selectivity(float quantile) : quantile(quantile) { assert((quantile >= 0) || (quantile == None.quantile)); }
		static const Selectivity None;
		static const Selectivity Infinit;

		[[nodiscard]] bool operator==(const Selectivity& o) const noexcept { return quantile == o.quantile; }
		[[nodiscard]] bool operator!=(const Selectivity& o) const noexcept { return quantile != o.quantile; }
		[[nodiscard]] bool operator<(const Selectivity& o) const noexcept { return quantile > o.quantile; }
		[[nodiscard]] bool operator>(const Selectivity& o) const noexcept { return quantile < o.quantile; }
		[[nodiscard]] bool operator<=(const Selectivity& o) const noexcept { return quantile >= o.quantile; }
		[[nodiscard]] bool operator>=(const Selectivity& o) const noexcept { return quantile <= o.quantile; }
	};

	[[nodiscard]] inline Selectivity min(Selectivity l, Selectivity r) noexcept { return (l < r) ? l : r; }
	[[nodiscard]] inline Selectivity max(Selectivity l, Selectivity r) noexcept { return (l < r) ? r : l; }

	struct Intensity
	{
		OpenInterval window;
		unsigned int depth;
		Selectivity selectivity;

		Intensity() = delete;
		Intensity(OpenInterval window, unsigned int depth, Selectivity selectivity)
			: window(window), depth(depth), selectivity(selectivity) {}

		static Intensity Exact(Position);

		[[nodiscard]] bool operator==(const Intensity& o) const noexcept { return (window == o.window) && (depth == o.depth) && (selectivity == o.selectivity); }
		[[nodiscard]] bool operator!=(const Intensity& o) const noexcept { return (window != o.window) || (depth != o.depth) || (selectivity != o.selectivity); }

		// Intensity with inverted window.
		[[nodiscard]] Intensity operator-() const; // TODO: Remove?

		// Subtracts depth.
		[[nodiscard]] Intensity operator-(int depth) const; // TODO: Remove?

		[[nodiscard]] Intensity next() const;
	};

	class Result
	{
	public:
		// TODO: Because members are public, the constraint can be violated.
		ClosedInterval window;
		unsigned int depth;
		Selectivity selectivity;
		Field best_move;
		std::size_t node_count;

		Result() = delete;
		Result(ClosedInterval window, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count);

		static Result ExactScore(Score, unsigned int depth, Selectivity, Field best_move, std::size_t node_count);
		static Result MaxBound(Score, unsigned int depth, Selectivity, Field best_move, std::size_t node_count);
		static Result MinBound(Score, unsigned int depth, Selectivity, Field best_move, std::size_t node_count);

		static Result ExactScore(Score, Intensity, Field best_move, std::size_t node_count);
		static Result MaxBound(Score, Intensity, Field best_move, std::size_t node_count);
		static Result MinBound(Score, Intensity, Field best_move, std::size_t node_count);

		// Result with inverted window.
		[[nodiscard]] Result operator-() const;
	};

	struct Algorithm
	{
		virtual Result Eval(Position, Intensity) = 0;
		Result Eval(Position pos) { return Eval(pos, Intensity::Exact(pos)); }
	};
}