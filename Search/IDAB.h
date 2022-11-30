#pragma once
#include "Core/Core.h"
#include "Algorithm.h"
#include "Intensity.h"
#include "Interval.h"
#include "MTDf.h"
#include <utility>

// Iterative Deepening And Broadening
template <typename Base>
class IDAB : public Algorithm
{
	Base base;
	static inline thread_local uint64 nodes;
public:
	template <typename... Args>
	IDAB(Args&&... args) noexcept : base(std::forward<Args>(args)...) {}

	uint64 Nodes() const override { return nodes; }

	using Algorithm::Eval;
	ContextualResult Eval(const Position& pos, Intensity request, OpenInterval window) override
	{
		nodes = 0;

		const int D = request.depth;
		const int E = pos.EmptyCount();
		std::vector<Confidence> confidence_levels;
		//if (D > 20)
		//	confidence_levels = { 1.1_sigmas, 1.5_sigmas, Confidence::Certain() }; // 1.1: 86%, 1.5: 93%, 2.0: 97%, 2.6: 99%
		//else
			confidence_levels = { 1.1_sigmas, /*1.5_sigmas,*/ Confidence::Certain() }; // 1.1: 86%, 1.5: 93%, 2.0: 97%, 2.6: 99%
		ContextualResult ret;


		// Iterative deepening
		for (int d = 0; d < D and d <= E - 10; d++)
		{
			ret = base.Eval(pos, { d, confidence_levels[0] }, window);
			nodes += base.Nodes();
		}

		// Iterative broadening
		for (int level = 0; confidence_levels[level] < request.certainty; level++)
		{
			auto mtd_f = MTD<Base>{ ret.score, base };
			ret = mtd_f.Eval(pos, { D, confidence_levels[level] }, window);
			nodes += mtd_f.Nodes();
			//ret = base.Eval(pos, { D, confidence_levels[level] }, window);
			//nodes += base.Nodes();
		}

		//return ret;
		
		//ret = base.Eval(pos, request);
		//nodes += base.Nodes();
		//return ret;

		auto mtd_f = MTD<Base>{ ret.score, base };
		ret = mtd_f.Eval(pos, request, window);
		nodes += mtd_f.Nodes();
		return ret;
	}

	void clear() override { base.clear(); }
};
