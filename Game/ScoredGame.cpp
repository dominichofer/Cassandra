#include "ScoredGame.h"
#include "Score.h"
#include <cassert>
#include <ranges>

ScoredGame::ScoredGame(Game game, std::vector<Score> scores) noexcept
	: game(std::move(game))
	, scores(std::move(scores))
{
	assert(this->scores.size() == this->game.Moves().size() + 1);
}

ScoredGame::ScoredGame(Game) noexcept
	: game(std::move(game))
	, scores(game.Moves().size() + 1, undefined_score)
{}

ScoredGame ScoredGame::FromString(std::string_view str)
{
	std::size_t spaces = std::ranges::count(str, ' ');

	// find the (space / 2 + 1)-th space
	auto it = std::begin(str);
	for (std::size_t i = 0; i < spaces / 2 + 1; ++i)
		it = std::ranges::find(it, std::end(str), ' ') + 1;

	std::string_view game_str{ std::begin(str), it - 1 };
	std::string_view scores_str{ it, std::end(str) };

	Game game = Game::FromString(game_str);

	std::vector<Score> scores;
	scores.reserve(game.Moves().size() + 1);
	for (auto word : std::ranges::split_view(scores_str, ' '))
		scores.push_back(Score::FromString(std::string_view{ std::begin(word), std::end(word) }));

	return ScoredGame{ std::move(game), std::move(scores) };
}

bool ScoredGame::operator==(const ScoredGame& o) const noexcept
{
	return game == o.game and scores == o.scores;
}

std::string to_string(const ScoredGame& gs)
{
	std::string game = to_string(gs.game);
	std::string scores = join(' ', gs.scores, [](Score s) { return to_string(s); });
	return game + " " + scores;
}