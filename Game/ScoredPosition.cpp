#include "ScoredPosition.h"

ScoredPosition ScoredPosition::FromString(std::string_view str)
{
	Position pos = Position::FromString(str);
	Score score = Score::FromString(str.substr(69));
	return { pos, score };
}

int ScoredPosition::EmptyCount() const noexcept
{
	return pos.EmptyCount();
}

bool ScoredPosition::HasScore() const noexcept
{
	return score != undefined_score;
}

std::string to_string(const ScoredPosition& ps)
{
	return to_string(ps.pos) + " % " + to_string(ps.score);
}
