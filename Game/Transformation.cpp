#include "Transformation.h"

std::vector<Position> Positions(std::vector<Game>::const_iterator begin, std::vector<Game>::const_iterator end)
{
	std::vector<Position> ret;
	for (auto it = begin; it != end; ++it)
		for (Position p : it->Positions())
			ret.push_back(p);
	return ret;
}

std::vector<Position> Positions(const std::vector<Game>& games)
{
	return Positions(games.begin(), games.end());
}

std::vector<PosScore> PosScoreFromGameScores(std::span<const GameScore> game_scores)
{
	std::vector<PosScore> ret;
	for (const GameScore& gs : game_scores)
	{
		auto pos = gs.game.Positions();
		for (int i = 0; i < pos.size(); i++)
			ret.emplace_back(pos[i], gs.scores[i]);
	}
	return ret;
}