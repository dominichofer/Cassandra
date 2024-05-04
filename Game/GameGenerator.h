#pragma once
#include "Board/Board.h"
#include "GameClass.h"
#include "Player.h"
#include <vector>

Game PlayedGame(Player& first, Player& second, Position start);
std::vector<Game> PlayedGames(Player& first, Player& second, const std::vector<Position>& starts);
