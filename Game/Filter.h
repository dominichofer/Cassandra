#pragma once
#include "Board/Board.h"
#include "ScoredPosition.h"
#include <ranges>
#include <span>
#include <vector>

std::vector<Position> EmptyCountFiltered(std::span<const Position>, int lower, int upper);
std::vector<Position> EmptyCountFiltered(std::span<const Position>, int empty_count);

std::vector<ScoredPosition> EmptyCountFiltered(std::span<const ScoredPosition>, int lower, int upper);
std::vector<ScoredPosition> EmptyCountFiltered(std::span<const ScoredPosition>, int empty_count);

auto FilterEmptyCount(int lower, int upper)
{
	return std::views::filter(
		[lower, upper](const auto& pos) { return lower <= pos.EmptyCount() and pos.EmptyCount() <= upper; }
	);
}

auto FilterEmptyCount(int empty_count)
{
	return std::views::filter(
		[empty_count](const auto& pos) { return pos.EmptyCount() == empty_count; }
	);
}
