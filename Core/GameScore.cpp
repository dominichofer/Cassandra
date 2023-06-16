#include "GameScore.h"
#include "Score.h"

GameScore::GameScore(Game game, std::vector<int> scores) noexcept
	: game(std::move(game))
	, scores(std::move(scores))
{
	if (this->scores.empty())
		clear_scores();
}

bool GameScore::operator==(const GameScore& o) const noexcept
{
	return game == o.game and scores == o.scores;
}

void GameScore::clear_scores()
{
	scores = std::vector<int>(game.Moves().size() + 1, 0);
}