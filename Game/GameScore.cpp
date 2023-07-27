#include "GameScore.h"
#include "Score.h"
#include <regex>

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
	scores = std::vector<int>(game.Moves().size() + 1, undefined_score);
}

bool IsGameScore(std::string_view str)
{
	std::regex pattern("[XO-]{64} [XO]( [A-H][1-8])*( [+-][0-9][0-9])*");
	return std::regex_match(str.begin(), str.end(), pattern);
}

std::string to_string(const GameScore& gs)
{
	std::string str = to_string(gs.game);
	for (int score : gs.scores)
		str += " " + ScoreToString(score);
	return str;
}

GameScore GameScoreFromString(std::string_view str)
{
	Position pos = PositionFromString(str);

	std::vector<Field> moves;
	std::vector<int> scores;
	int i = 67;
	while (i < str.length())
		if (str[i] == '+' or str[i] == '-')
		{
			scores.push_back(ScoreFromString(str.substr(i, 3)));
			i += 4;
		}
		else
		{
			moves.push_back(FieldFromString(str.substr(i, 2)));
			i += 3;
		}

	return GameScore(Game(pos, std::move(moves)), std::move(scores));
}