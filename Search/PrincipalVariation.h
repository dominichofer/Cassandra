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
	using AlphaBetaFailSuperSoft::Eval_BestMove;
	int Eval(const Position&, Intensity, OpenInterval) override;
	ScoreMove Eval_BestMove(const Position&, Intensity, OpenInterval) override;

	void clear() override;
private:
	ScoreMove Eval_BestMove_N(const Position&, Intensity, OpenInterval);
protected:
	IntensityScore PVS_N(const Position&, Intensity, const OpenInterval&);
	IntensityScore ZWS_N(const Position&, Intensity, const OpenInterval&);
private:
	std::optional<IntensityScore> MPC(const Position&, Intensity, const OpenInterval&);
	IntensityScore Eval_dN(const Position&, Intensity, OpenInterval);
	IntensityScore Eval_d0(const Position&);

	SortedMoves SortMoves(Moves, const Position&);
};