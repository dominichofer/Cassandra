#pragma once
#include "Position.h"
#include "Moves.h"

#include <cassert>
#include <cstdint>
#include <cmath>
#include <chrono>

using Score = int;

Score EvalGameOver(Position);

namespace Search
{
	constexpr Score infinity = +65;

	class Window
	{
	public:
		Window() = default;
		Window(Score alpha, Score beta) : alpha(alpha), beta(beta) { assert(alpha < beta); }
		
		Score alpha{ -infinity }, beta{ +infinity };

		Window operator-() const { return { -beta, -alpha }; }
	};
	
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

		explicit Selectivity(float quantile);
		static const Selectivity None;
	};

	struct Intensity
	{
		unsigned int depth;
		Selectivity selectivity;
		Window window;

		static Intensity Exact(Position);
	};

	struct Result
	{
		Score score;
		Field best_move;
		std::size_t node_count;
	};

	struct Algorithm
	{
		virtual Result Eval(Position, Intensity) = 0;
	};
}