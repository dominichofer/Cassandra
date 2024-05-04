#pragma once
#include "Board/Board.h"
#include "ScoredPosition.h"
#include "GameClass.h"
#include "ScoredGame.h"
#include <vector>

std::vector<Position> Positions(const Game&);
std::vector<Position> Positions(const ScoredGame&);
std::vector<Position> Positions(std::span<const Game>);
std::vector<Position> Positions(std::span<const ScoredGame>);
std::vector<Position> Positions(std::span<const ScoredPosition>);
std::vector<Score> Scores(const ScoredGame&);
std::vector<Score> Scores(std::span<const ScoredPosition>);

std::vector<ScoredPosition> ScoredPositions(std::span<const Game>);
std::vector<ScoredPosition> ScoredPositions(std::span<const ScoredGame>);
std::vector<ScoredPosition> ScoredPositions(std::span<const Position>, std::span<const Score>);
