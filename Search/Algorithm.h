#pragma once
#include "Game/Game.h"
#include "Estimator.h"
#include "HashTable.h"
#include "Result.h"
#include "MoveSorter.h"
#include <atomic>
#include <cstdint>
#include <optional>

class Algorithm
{
public:
	virtual Result Eval(int guess, const Position&, OpenInterval window, Intensity);
	virtual Result Eval(const Position&, OpenInterval window, Intensity) = 0;
	Result Eval(const Position& pos, Intensity intensity) { return Eval(pos, { min_score, max_score }, intensity); }
	Result Eval(const Position& pos, OpenInterval window) { return Eval(pos, window, pos.EmptyCount()); }
	Result Eval(const Position& pos) { return Eval(pos, pos.EmptyCount()); }

	virtual uint64_t& Nodes() noexcept = 0;
	virtual void Clear() noexcept = 0;
};

class NegaMax : public Algorithm
{
public:
	static inline thread_local uint64_t nodes{ 0 };

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, Intensity) override;
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
	Result Eval(const Position&, OpenInterval window, Intensity) override;
protected:
	Score Eval_N(const Position&, OpenInterval window);
	Score Eval_P(const Position&, OpenInterval window);
	Score Eval_3(const Position&, OpenInterval window, Field, Field, Field);
	Score Eval_2(const Position&, const OpenInterval& window, Field, Field);
};

class PVS : public AlphaBeta
{
protected:
	HashTable& tt;
	const Estimator& estimator;
	std::atomic_flag stop;
	//PVS* parallel_nodes = nullptr;
public:
	PVS(HashTable&, const Estimator&) noexcept;
	bool Stop() { /*if (parallel_nodes) parallel_nodes->Stop();*/ return stop.test_and_set(std::memory_order_acq_rel); }
	bool IsStop() { return stop.test(std::memory_order_acquire); }

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, Intensity) override;
	void Clear() noexcept override { AlphaBeta::Clear(); tt.Clear(); }
private:
	SortedMoves Sorted(const Position&, Intensity);
	Result PVS_N(const Position&, OpenInterval window, Intensity);
	Result ZWS_N(const Position&, OpenInterval window, Intensity);
	Result Parallel_ZWS_N(const Position&, OpenInterval window, Intensity);

	Result Eval_dN(const Position&, OpenInterval window, int depth);
	Result Eval_d0(const Position&);
	std::optional<Result> TTC(const Position&, OpenInterval window, Intensity);
	std::optional<Result> ETC(const Position&, OpenInterval window, Intensity);
	std::optional<Result> MPC(const Position&, OpenInterval window, Intensity);
	void InsertTT(const Position&, const Result&);
};

// Iterative Deepening And Broadening
class IDAB : public Algorithm
{
	Algorithm& alg;
public:
	IDAB(Algorithm& alg) noexcept : alg(alg) {}

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, Intensity) override;

	uint64_t& Nodes() noexcept override { return alg.Nodes(); }
	void Clear() noexcept override { alg.Clear(); }
};

// Parallel Iterative Deepening And Broadening
class PIDAB : public Algorithm
{
	Algorithm& alg;
public:
	PIDAB(Algorithm& alg) noexcept : alg(alg) {}

	using Algorithm::Eval;
	Result Eval(const Position&, OpenInterval window, Intensity) override;

	uint64_t& Nodes() noexcept override { return alg.Nodes(); }
	void Clear() noexcept override { alg.Clear(); }
};

// Memory-enhanced Test Driver
class MTD : public Algorithm
{
	Algorithm& alg;
public:
	MTD(Algorithm& alg) noexcept : alg(alg) {}

	using Algorithm::Eval;
	Result Eval(int guess, const Position&, OpenInterval window, Intensity) override;
	Result Eval(const Position&, OpenInterval window, Intensity) override;

	uint64_t& Nodes() noexcept override { return alg.Nodes(); }
	void Clear() noexcept override { alg.Clear(); }
};

// Aspiration Search
class AspirationSearch : public Algorithm
{
	Algorithm& alg;
public:
	AspirationSearch(Algorithm& alg) noexcept : alg(alg) {}

	using Algorithm::Eval;
	Result Eval(int guess, const Position&, OpenInterval window, Intensity) override;
	Result Eval(const Position&, OpenInterval window, Intensity) override;

	uint64_t& Nodes() noexcept override { return alg.Nodes(); }
	void Clear() noexcept override { alg.Clear(); }
};
