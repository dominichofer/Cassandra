#include "AlphaBetaFailHardSearch.h"
#include "Core/Machine.h"
#include <algorithm>

using namespace Search;

Score clamp(Score value, OpenInterval w)
{
	return std::clamp(value, w.lower, w.upper);
}

Result AlphaBetaFailHard::Eval(Position pos, Intensity intensity)
{
	node_counter = 0;
	Score score = Eval_triage(pos, intensity.window);
	return Result::ExactScore(score, pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
}

Score AlphaBetaFailHard::Eval_triage(const Position& pos, OpenInterval w)
{
	auto moves = Moves(pos.Empties());
	const auto move1 = moves.front(); moves.pop_front();
	const auto move2 = moves.front(); moves.pop_front();
	const auto move3 = moves.front(); moves.pop_front();
	const auto move4 = moves.front(); moves.pop_front();
	switch (pos.EmptyCount())
	{
		case 0: return Eval_0(pos, w);
		case 1: return Eval_1(pos, w, move1);
		case 2: return Eval_2(pos, w, move1, move2);
		case 3: return Eval_3(pos, w, move1, move2, move3);
		case 4: return Eval_4(pos, w, move1, move2, move3, move4);
		default: return Eval_N(pos, w);
	}
}

Score AlphaBetaFailHard::Eval_0(const Position& pos, const OpenInterval w)
{
	return clamp(NegaMax::Eval_0(pos), w);
}

Score AlphaBetaFailHard::Eval_1(const Position& pos, const OpenInterval w, const Field move1)
{
	return clamp(NegaMax::Eval_1(pos, move1), w);
}

Score AlphaBetaFailHard::Eval_2(const Position& pos, OpenInterval w, const Field move1, const Field move2)
{
	node_counter++;
	Score score = Score::Infinity;

	if (const auto flips = Flips(pos, move1)) {
		score = -Eval_1(Play(pos, move1, flips), -w, move2);
		if (score >= w.upper)
			return w.upper;
		w.lower = std::max(w.lower, score);
	}

	if (const auto flips = Flips(pos, move2)) {
		score = -Eval_1(Play(pos, move2, flips), -w, move1);
		if (score >= w.upper)
			return w.upper;
		return std::max(score, w.lower);
	}

	if (score != Score::Infinity)
		return w.lower;

	const auto passed = PlayPass(pos);
	node_counter++;

	if (const auto flips = Flips(passed, move1)) {
		score = Eval_1(Play(passed, move1, flips), w, move2);
		if (score <= w.lower)
			return w.lower;
		w.upper = std::min(w.upper, score);
	}

	if (const auto flips = Flips(passed, move2)) {
		score = Eval_1(Play(passed, move2, flips), w, move1);
		if (score <= w.lower)
			return w.lower;
		return std::min(score, w.upper);
	}

	if (score != Score::Infinity)
		return w.upper;

	node_counter--;
	return clamp(-EvalGameOver(passed), w);
}

Score AlphaBetaFailHard::Eval_3(const Position& pos, OpenInterval w, const Field move1, const Field move2, const Field move3)
{
	node_counter++;
	Score score = Score::Infinity;

	if (const auto flips = Flips(pos, move1)) {
		score = -Eval_2(Play(pos, move1, flips), -w, move2, move3);
		if (score >= w.upper)
			return w.upper;
		w.lower = std::max(w.lower, score);
	}

	if (const auto flips = Flips(pos, move2)) {
		score = -Eval_2(Play(pos, move2, flips), -w, move1, move3);
		if (score >= w.upper)
			return w.upper;
		w.lower = std::max(w.lower, score);
	}

	if (const auto flips = Flips(pos, move3)) {
		score = -Eval_2(Play(pos, move3, flips), -w, move1, move2);
		if (score >= w.upper)
			return w.upper;
		return std::max(score, w.lower);
	}

	if (score != Score::Infinity)
		return w.lower;

	const auto passed = PlayPass(pos);
	node_counter++;

	if (const auto flips = Flips(passed, move1)) {
		score = Eval_2(Play(passed, move1, flips), w, move2, move3);
		if (score <= w.lower)
			return w.lower;
		w.upper = std::min(w.upper, score);
	}

	if (const auto flips = Flips(passed, move2)) {
		score = Eval_2(Play(passed, move2, flips), w, move1, move3);
		if (score <= w.lower)
			return w.lower;
		w.upper = std::min(w.upper, score);
	}

	if (const auto flips = Flips(passed, move3)) {
		score = Eval_2(Play(passed, move3, flips), w, move1, move2);
		if (score <= w.lower)
			return w.lower;
		return std::min(score, w.upper);
	}

	if (score != Score::Infinity)
		return w.upper;

	node_counter--;
	return clamp(-EvalGameOver(passed), w);
}

Score AlphaBetaFailHard::Eval_4(const Position& pos, OpenInterval w, const Field move1, const Field move2, const Field move3, const Field move4)
{
	node_counter++;
	Score score = Score::Infinity;

	if (const auto flips = Flips(pos, move1)) {
		score = -Eval_3(Play(pos, move1, flips), -w, move2, move3, move4);
		if (score >= w.upper)
			return w.upper;
		w.lower = std::max(w.lower, score);
	}

	if (const auto flips = Flips(pos, move2)) {
		score = -Eval_3(Play(pos, move2, flips), -w, move1, move3, move4);
		if (score >= w.upper)
			return w.upper;
		w.lower = std::max(w.lower, score);
	}

	if (const auto flips = Flips(pos, move3)) {
		score = -Eval_3(Play(pos, move3, flips), -w, move1, move2, move4);
		if (score >= w.upper)
			return w.upper;
		w.lower = std::max(w.lower, score);
	}

	if (const auto flips = Flips(pos, move4)) {
		score = -Eval_3(Play(pos, move4, flips), -w, move1, move2, move3);
		if (score >= w.upper)
			return w.upper;
		return std::max(score, w.lower);
	}

	if (score != Score::Infinity)
		return w.lower;

	const auto passed = PlayPass(pos);
	node_counter++;

	if (const auto flips = Flips(passed, move1)) {
		score = Eval_3(Play(passed, move1, flips), w, move2, move3, move4);
		if (score <= w.lower)
			return w.lower;
		w.upper = std::min(w.upper, score);
	}

	if (const auto flips = Flips(passed, move2)) {
		score = Eval_3(Play(passed, move2, flips), w, move1, move3, move4);
		if (score <= w.lower)
			return w.lower;
		w.upper = std::min(w.upper, score);
	}

	if (const auto flips = Flips(passed, move3)) {
		score = Eval_3(Play(passed, move3, flips), w, move1, move2, move4);
		if (score <= w.lower)
			return w.lower;
		w.upper = std::min(w.upper, score);
	}

	if (const auto flips = Flips(passed, move4)) {
		score = Eval_3(Play(passed, move4, flips), w, move1, move2, move3);
		if (score <= w.lower)
			return w.lower;
		return std::min(w.upper, score);
	}

	if (score != Score::Infinity)
		return w.upper;
	
	node_counter--;
	return clamp(-EvalGameOver(passed), w);
}

Score AlphaBetaFailHard::Eval_N(const Position& pos, OpenInterval w)
{
	if (pos.EmptyCount() <= 4)
		return Eval_triage(pos, w);

	node_counter++;

	Moves moves = PossibleMoves(pos);
	if (moves.empty()) {
		const auto passed = PlayPass(pos);
		if (PossibleMoves(passed).empty())
			return clamp(EvalGameOver(pos), w);
		return -Eval_N(passed, -w);
	}

	for (auto move : moves)
	{
		const auto score = -Eval_N(Play(pos, move), -w);
		if (score >= w.upper)
			return w.upper;
		w.lower = std::max(w.lower, score);
	}

	return w.lower;
}

