#include "Transformation.h"
#include <cassert>

std::vector<Position> Positions(const Game& game)
{
	return game.Positions();
}

std::vector<Position> Positions(const ScoredGame& scored_game)
{
	return scored_game.game.Positions();
}

std::vector<Position> Positions(std::span<const Game> games)
{
	std::vector<Position> pos;
	for (auto it = games.begin(); it != games.end(); ++it)
		for (Position p : it->Positions())
			pos.push_back(p);
	return pos;
}

std::vector<Position> Positions(std::span<const ScoredGame> scored_games)
{
	std::vector<Position> pos;
	for (auto it = scored_games.begin(); it != scored_games.end(); ++it)
		for (Position p : it->game.Positions())
			pos.push_back(p);
	return pos;
}

std::vector<Position> Positions(std::span<const ScoredPosition> scored_pos)
{
	std::vector<Position> pos;
	for (auto it = scored_pos.begin(); it != scored_pos.end(); ++it)
		pos.push_back(it->pos);
	return pos;
}

std::vector<Score> Scores(const ScoredGame& scored_game)
{
	return scored_game.scores;
}

std::vector<Score> Scores(std::span<const ScoredPosition> scored_pos)
{
	std::vector<Score> scores;
	for (auto it = scored_pos.begin(); it != scored_pos.end(); ++it)
		scores.push_back(it->score);
	return scores;
}

std::vector<ScoredPosition> ScoredPositions(std::span<const Game> game)
{
	std::vector<ScoredPosition> ret;
	for (auto it = game.begin(); it != game.end(); ++it)
	{
		auto pos = it->Positions();
		for (int i = 0; i < pos.size(); i++)
			ret.emplace_back(pos[i]);
	}
	return ret;
}

std::vector<ScoredPosition> ScoredPositions(std::span<const ScoredGame> scored_games)
{
	std::vector<ScoredPosition> ret;
	for (auto it = scored_games.begin(); it != scored_games.end(); ++it)
	{
		auto pos = it->game.Positions();
		for (int i = 0; i < pos.size(); i++)
			ret.emplace_back(pos[i], it->scores[i]);
	}
	return ret;
}

std::vector<ScoredPosition> ScoredPositions(std::span<const Position> pos, std::span<const Score> scores)
{
	assert(pos.size() == scores.size());
	std::vector<ScoredPosition> ret;
	for (int i = 0; i < pos.size(); i++)
		ret.emplace_back(pos[i], scores[i]);
	return ret;
}
