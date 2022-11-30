#pragma once
#include "Game.h"
#include "Player.h"
#include "Position.h"
#include <random>
#include <vector>

Game PlayedGame(Player& first, Player& second, Position start);
std::vector<Game> PlayedGamesFrom(Player& first, Player& second, const std::vector<Position>& starts); 

Game SelfPlayedGame(Player&, Position start);
std::vector<Game> SelfPlayedGamesFrom(Player&, const std::vector<Position>& starts);

Game RandomGame(Position start, unsigned int seed = std::random_device{}());
std::vector<Game> RandomGamesFrom(const std::vector<Position>& starts, unsigned int seed = std::random_device{}());
