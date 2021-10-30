#pragma once
#include "Core/Core.h"
#include "HashTablePVS.h"
#include "Pattern/Evaluator.h"

class FixedDepthPlayer final : public Player
{
	HashTablePVS& tt;
	AAGLEM& evaluator;
	int depth;
	std::mt19937_64 rnd_engine;
public:
	FixedDepthPlayer(HashTablePVS& tt, AAGLEM& evaluator, int depth, uint64_t seed = std::random_device{}()) : tt(tt), evaluator(evaluator), depth(depth), rnd_engine(seed) {}

	Position Play(const Position&) override;
};