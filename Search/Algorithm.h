#pragma once
#include "Board/Board.h"
#include "Estimator.h"
#include "HashTable.h"
#include "Result.h"
#include "SortedMoves.h"
#include <cstdint>
#include <string>


class Algorithm
{
public:
	virtual Result Eval(const Position&, OpenInterval window, int depth, float confidence_level) = 0;
	Result Eval(const Position& pos) { return Eval(pos, { -inf_score, +inf_score }, pos.EmptyCount(), inf); }

	virtual uint64_t& Nodes() noexcept = 0;
	virtual void Clear() noexcept = 0;
};


class NegaMax : public Algorithm
{
public:
	static inline thread_local uint64_t nodes{ 0 };

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, int depth, float confidence_level) override;
	uint64_t& Nodes() noexcept override { return nodes; }
	void Clear() noexcept override { nodes = 0; }
protected:
	int Eval_N(const Position&);
	int Eval_3(const Position&, Field, Field, Field);
	int Eval_2(const Position&, Field, Field);
	int Eval_1(const Position&, Field);
	int Eval_0(const Position&);
};


// Alaph Beta fail soft
class AlphaBeta : public NegaMax
{
public:
	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, int depth, float confidence_level) override;
protected:
	int Eval_N(const Position&, OpenInterval window);
	int Eval_P(const Position&, OpenInterval window);
	int Eval_3(const Position&, OpenInterval window, Field, Field, Field);
	int Eval_2(const Position&, OpenInterval window, Field, Field);
};


class PVS : public AlphaBeta
{
protected:
	HT& tt;
	const Estimator& estimator;
public:
	PVS(HT&, const Estimator&) noexcept;

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, int depth, float confidence_level) override;
	void Clear() noexcept override { AlphaBeta::Clear(); tt.clear(); }
protected:
	Result PVS_N(const Position&, OpenInterval window, int depth, float confidence_level);
	Result ZWS_N(const Position&, OpenInterval window, int depth, float confidence_level);
private:
	Result Eval_dN(const Position&, OpenInterval window, int depth);
	Result Eval_d0(const Position&);
	Result EndScore(const Position&);
	SortedMoves Sorted(const Position&, int depth, float confidence_level);
	std::optional<Result> TTC(const Position&, OpenInterval window, int depth, float confidence_level);
	std::optional<Result> ETC(const Position&, OpenInterval window, int depth, float confidence_level);
	std::optional<Result> MPC(const Position&, OpenInterval window, int depth, float confidence_level);
};


// Memory-enhanced Test Driver
class MTD : public Algorithm
{
protected:
	Algorithm& alg;
public:
	MTD(Algorithm& alg) noexcept : alg(alg) {}

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, int depth, float confidence_level) override;
	Result Eval(int guess, const Position&, OpenInterval window, int depth, float confidence_level);
	uint64_t& Nodes() noexcept override { return alg.Nodes(); }
	void Clear() noexcept override { alg.Clear(); }
};


// Iterative Deepening And Broadening
class IDAB : public MTD
{
public:
	IDAB(Algorithm& alg) noexcept : MTD(alg) {}

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, int depth, float confidence_level) override;
};
