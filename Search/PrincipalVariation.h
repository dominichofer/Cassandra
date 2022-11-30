#pragma once
#include "Core/Core.h"
#include "Pattern/Pattern.h"
#include "Algorithm.h"
#include "AlphaBetaFailHard.h"
#include "AlphaBetaFailSoft.h"
#include "AlphaBetaFailSuperSoft.h"
#include "HashTable.h"
#include "SortedMoves.h"
#include <optional>

class PVS : public AlphaBetaFailSuperSoft
{
protected:
	HT& tt;
	const AAGLEM& evaluator;
public:
	static inline uint64_t counter = 0; // TODO: Remove!
	PVS(HT& tt, const AAGLEM& evaluator) noexcept : tt(tt), evaluator(evaluator) {}

	using AlphaBetaFailSuperSoft::Eval;
	ContextualResult Eval(const Position&, Intensity, OpenInterval) override;

	void clear() override;
protected:
	ContextualResult PVS_N(const Position&, Intensity, const OpenInterval&);
	ContextualResult ZWS_N(const Position&, Intensity, const OpenInterval&);
private:
	std::optional<ContextualResult> MPC(const Position&, Intensity, const OpenInterval&);
	ContextualResult Eval_dN(const Position&, Intensity, OpenInterval);
	ContextualResult Eval_d0(const Position&);

	SortedMoves SortMoves(Moves, const Position&, int depth);
};