#pragma once
#include "Core/Core.h"
#include "AlphaBeta.h"
#include "HashTable.h"
#include "Result.h"
#include "SortedMoves.h"
#include <optional>

class Status
{
	int alpha;
	int best_score;
	Field best_move;
	float worst_confidence_level;
	int smallest_depth;
public:
	Status(int alpha);

	void Update(Result, Field move);
	Result GetResult();
};

class PVS : public AlphaBeta
{
protected:
	HT& tt;
	const Estimator& estimator;
public:
	PVS(HT&, const Estimator&) noexcept;

	ResultTimeNodes Eval(const Position&);
	ResultTimeNodes Eval(const Position&, OpenInterval window, int depth, float confidence_level);
protected:
	Result PVS_N(const Position&, OpenInterval window, int depth, float confidence_level);
	Result ZWS_N(const Position&, OpenInterval window, int depth, float confidence_level);
private:
	Result EndScore(const Position&);
	Result Eval_d1(const Position&);
	Result Eval_d0(const Position&);
	SortedMoves Sorted(const Position&, OpenInterval window, int depth, float confidence_level);
	std::optional<Result> MPC(const Position&, OpenInterval window, int depth, float confidence_level);
};
