#pragma once
#include "Board/Board.h"
#include "PositionScore.h"
#include <span>
#include <vector>

std::vector<Position> EmptyCountFiltered(std::span<const Position>, int min_empty_count, int max_empty_count);
std::vector<Position> EmptyCountFiltered(std::span<const Position>, int empty_count);

std::vector<PosScore> EmptyCountFiltered(std::span<const PosScore>, int min_empty_count, int max_empty_count);
std::vector<PosScore> EmptyCountFiltered(std::span<const PosScore>, int empty_count);
