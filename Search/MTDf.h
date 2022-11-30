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

	ContextualResult Eval(const Position& pos, Intensity request, OpenInterval window) override
	{
		nodes = 0;

		// From https://en.wikipedia.org/wiki/MTD(f)
		ContextualResult g{ guess };
		int upperBound = window.Upper();
		int lowerBound = window.Lower();

		while (lowerBound < upperBound)
		{
			int beta = std::max<int>(g, lowerBound + 1);

			g = base.Eval(pos, request, { beta - 1, beta });

			nodes += base.Nodes();

			if (g.score < beta)
				upperBound = g.score;
			else
				lowerBound = g.score;
		}
		return g;
	}
};
