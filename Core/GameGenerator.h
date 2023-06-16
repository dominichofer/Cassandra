#pragma once
#include "Game.h"
#include "Player.h"
#include "Position.h"
#include <vector>

Game PlayedGame(Player& first, Player& second, Position start);
std::vector<Game> PlayedGamesFrom(Player& first, Player& second, const std::vector<Position>& starts);
