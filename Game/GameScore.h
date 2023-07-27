#pragma once
#include "Board/Board.h"
#include "GameClass.h"
#include <string>
#include <string_view>
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

bool IsGameScore(std::string_view);
std::string to_string(const GameScore&);
GameScore GameScoreFromString(std::string_view);
