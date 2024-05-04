//#pragma once
//#include "Algorithm.h"
//#include "MoveSorter.h"
//
//// Dynamic Tree Splitting
//class DTS : public Algorithm
//{
//protected:
//	HT& tt;
//	Algorithm& alg;
//	MoveSorter& move_sorter;
//	int parallel_plies;
//public:
//	DTS(HT& tt, Algorithm& alg, MoveSorter& move_sorter, int parallel_plies) noexcept;
//
//	using Algorithm::Eval;
//	Result Eval(const Position&, OpenInterval window, Intensity intensity) override;
//	uint64_t& Nodes() noexcept override { return alg.Nodes(); }
//	void Clear() noexcept override { alg.Clear(); }
//};
