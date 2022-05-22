#pragma once
#include "Core/Core.h"
#include "Algorithm.h"
#include "Intensity.h"
#include "Interval.h"
#include <utility>

template <typename Base>
class MTD : public Algorithm
{
	const int guess;
	Base base;
	static inline thread_local uint64 nodes;
public:
	template <typename... Args>
	MTD(int guess, Args&&... args) noexcept : guess(guess), base(std::forward<Args>(args)...) {}

	uint64 Nodes() const override { return nodes; }

	int Eval(const Position& pos, Intensity i, OpenInterval w) override { return eval(pos, i, w, false); }
	ScoreMove Eval_BestMove(const Position& pos, Intensity i, OpenInterval w) override { return eval(pos, i, w, true); }
private:
	ScoreMove eval(const Position& pos, Intensity request, OpenInterval window, bool best_move)
	{
		nodes = 0;

		// From https://en.wikipedia.org/wiki/MTD(f)
		ScoreMove g = guess;
		int upperBound = window.Upper();
		int lowerBound = window.Lower();

		while (lowerBound < upperBound)
		{
			int beta = std::max<int>(g, lowerBound + 1);

			if (best_move)
				g = base.Eval_BestMove(pos, request, { beta - 1, beta });
			else
				g = base.Eval(pos, request, { beta - 1, beta });

			nodes += base.Nodes();

			if (g < beta)
				upperBound = g;
			else
				lowerBound = g;
		}
		return g;
	}
};
