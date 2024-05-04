#pragma once
#include "Board/Board.h"
#include "GameClass.h"
#include "Score.h"
#include <string>
#include <string_view>
#include <vector>

class ScoredGame
{
public:
    Game game;
    std::vector<Score> scores;

    ScoredGame(Game) noexcept;
    ScoredGame(Game, std::vector<Score> scores) noexcept;
    static ScoredGame FromString(std::string_view);

    bool operator==(const ScoredGame&) const noexcept;
    bool operator!=(const ScoredGame& o) const noexcept { return !(*this == o); }
};

std::string to_string(const ScoredGame&);
