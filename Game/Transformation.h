#pragma once
#include "Board/Board.h"
#include "PositionScore.h"
#include "GameClass.h"
#include "GameScore.h"
#include <span>
#include <vector>

std::vector<Position> Positions(std::vector<Game>::const_iterator begin, std::vector<Game>::const_iterator end);
std::vector<Position> Positions(const std::vector<Game>&);

std::vector<PosScore> PosScoreFromGameScores(std::span<const GameScore>);
