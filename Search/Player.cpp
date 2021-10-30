#include "Player.h"
#include "Algorithm.h"
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>

Position FixedDepthPlayer::Play(const Position& pos)
{
	auto possible_moves = PossibleMoves(pos);
	if (not possible_moves)
		return PlayPass(pos);
	//return ::Play(pos, PVS{ tt, evaluator }.BestMove(pos, Intensity(depth - 1)));
	std::vector<std::pair<Field, int>> move_score;
	for (Field move : possible_moves)
		move_score.emplace_back(move, PVS{ tt, evaluator }.Eval(::Play(pos, move), Intensity(depth - 1)));
	std::shuffle(move_score.begin(), move_score.end(), rnd_engine);
	std::sort(move_score.begin(), move_score.end(), [](const std::pair<Field, int>& l, const std::pair<Field, int>& r) { return l.second < r.second; });
	return ::Play(pos, move_score.back().first);
}
