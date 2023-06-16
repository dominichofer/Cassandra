#pragma once
#include "Game.h"
#include <vector>

class GameScore
{
public:
    Game game;
    std::vector<int> scores;

    GameScore(Game, std::vector<int> scores) noexcept;

    bool operator==(const GameScore&) const noexcept;

    void clear_scores();
};
