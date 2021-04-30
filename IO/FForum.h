#pragma once
#include "Search/Puzzle.h"
#include <vector>

static std::vector<Puzzle> FForum =
{
	Puzzle::Exact(Position::Start(), +0),
	Puzzle::Exact(
		"- - X X X X X -"
		"- O O O X X - O"
		"- O O O X X O X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- X X X O O O -"
		"- O O O O O - -"_pos, +18 / 2 // 01
	),
	Puzzle::Exact(
		"- X X X X X X -"
		"- - X O O O O -"
		"- X O X X O O X"
		"X O O O O O O O"
		"O X O O X X O O"
		"O O O X X O O X"
		"- O O O O O - -"
		"- - X X X X X -"_pos, +10 / 2 // 02
	),
	Puzzle::Exact(
		"- - - - O X - -"
		"- - O O X X - -"
		"- O O O X X - X"
		"O O X X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X X O X O"
		"- - O O O O O X"_pos, +2 / 2 // 03
	),
	Puzzle::Exact(
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X X O X O O O X"
		"- O X O O X X X"
		"- - O O O X X X"
		"- - O O X X - -"
		"- - X O X X O -"_pos, +0 / 2 // 04
	),
	Puzzle::Exact(
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O X O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O O - -"
		"- X X X X X - -"_pos, +32 / 2 // 05
	),
	Puzzle::Exact(
		"- - O X X X - -"
		"O O O X X X - -"
		"O O O X O X O -"
		"O O X O O O X -"
		"O O X X X X X X"
		"X O O X X O X -"
		"- O O O O X - -"
		"- X X X X X X -"_pos, +14 / 2 // 06
	),
	Puzzle::Exact(
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O X X X X"
		"X O O X X X X X"
		"X O O O O X X X"
		"- X X X X X X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 07
	),
	Puzzle::Exact(
		"- - - O - O - -"
		"O - O O O O - -"
		"O O O O X O O O"
		"O O O X X X X X"
		"O O X O O O X -"
		"O X O O O O X -"
		"O X X O O O - -"
		"O X X O O X - -"_pos, +8 / 2 // 08
	),
	Puzzle::Exact(
		"- - O X O O - -"
		"X - X X O O O O"
		"- X X X O O O O"
		"- O X O O O X O"
		"O O X O X X X O"
		"X O O X O X O O"
		"- - X O X X - -"
		"- - X X X X - -"_pos, -8 / 2 // 09
	),
	Puzzle::Exact(
		"- O O O O - - -"
		"- - X O O O - -"
		"O X O X O X O O"
		"X O X O O X O O"
		"X O O X O X X X"
		"O O O X O X X O"
		"- - X O O X - -"
		"- X X X X X - -"_pos, +10 / 2 // 10
	),
	Puzzle::Exact(
		"- - - X - O X O"
		"- - - - O O X O"
		"- - - O O X X O"
		"X - O O X O X O"
		"O O O X X O X O"
		"- O X X O O O O"
		"O X X X O O - O"
		"X X X X X X X -"_pos, +30 / 2 // 11
	),
	Puzzle::Exact(
		"- - X - - X - -"
		"O - X X X X O -"
		"O O X X X O X X"
		"O O X O X O X X"
		"O O X O O X X X"
		"O O O O X X X X"
		"- - X O O O - -"
		"- O O O O O - -"_pos, -8 / 2 // 12
	),
	Puzzle::Exact(
		"- - X X X X X -"
		"- O O O X X - -"
		"- O O O X X X X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- - O X O O O -"
		"- O O O O O - -"_pos, +14 / 2 // 13
	),
	Puzzle::Exact(
		"- - X X X X X -"
		"- - O O O X - -"
		"- X O O X X X X"
		"- O O O O O O O"
		"O O O X X X O O"
		"O O O X X O O X"
		"- - X X O O - -"
		"- - X X X X X -"_pos, +18 / 2 // 14
	),
	Puzzle::Exact(
		"- - - - O - - -"
		"- - - O O X - -"
		"- O O O X X - X"
		"O O O X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X O O X O"
		"- - O O O O O X"_pos, +4 / 2 // 15
	),
	Puzzle::Exact(
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X O O X X X O X"
		"- O O O X X X X"
		"- - O O X X X X"
		"- - - O O O - -"
		"- - X O X - O -"_pos, +24 / 2 // 16
	),
	Puzzle::Exact(
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O O O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O - - -"
		"- X X X X - - -"_pos, +8 / 2 // 17
	),
	Puzzle::Exact(
		"- X X X - - - -"
		"- - O O O X - -"
		"X O O O O O X X"
		"O X O X O O X X"
		"O X X O O O O O"
		"X X X O X O O X"
		"- - O X X O - -"
		"- O O O O O - -"_pos, -2 / 2 // 18
	),
	Puzzle::Exact(
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O O X X X"
		"X O O O X X X X"
		"X - O O O X X X"
		"- - O O O O X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 19
	),
	Puzzle::Exact(
		"X X X O X X X X"
		"O X X X X X X X"
		"O O X X X X X X"
		"O O O X X X X X"
		"O O O X X O O -"
		"O O O O O - - -"
		"O O O O O O O -"
		"O O O O O O O -"_pos, +6 / 2 // 20
	),
	Puzzle::Exact(
		"X X X X X X X X"
		"O X X O O O - -"
		"O O X X O X X -"
		"O X O X X X - -"
		"O X X X X O - -"
		"O X X O X X - -"
		"O X X X X X - -"
		"O O O O - - - -"_pos, +0 / 2 // 21
	),
	Puzzle::Exact(
		"- - X X X X - -"
		"O - X X X X X -"
		"O O X X O X O O"
		"O X O X O O O O"
		"O O O X O O O O"
		"- O O X O X O O"
		"- - X O O O - O"
		"- - - - O - - -"_pos, +2 / 2 // 22
	),
	Puzzle::Exact(
		"- - O - - - - -"
		"- - O O X - - -"
		"O O O X X X O -"
		"O O O O X O X X"
		"X X X O O X O X"
		"X X X X X O O X"
		"X - X X X X O X"
		"- - X X X X - -"_pos, +4 / 2 // 23
	),
	Puzzle::Exact(
		"- - X - - X - -"
		"- - - X X X O -"
		"- O - O X O X X"
		"- - O O O X X X"
		"O O O O X X X X"
		"O O O X O O X X"
		"O O O O O O - -"
		"O X O O - X - -"_pos, +0 / 2 // 24
	),
	Puzzle::Exact(
		"- - - - O - - -"
		"- - - O O O X -"
		"- X X X O O O O"
		"O X X X X O O X"
		"- O O X X O O X"
		"X X O X O O O O"
		"X X X O O - - -"
		"O - O O O O - -"_pos, +0 / 2 // 25
	),
	Puzzle::Exact(
		"- O O O O O - -"
		"- - O X X O - -"
		"- O O O O X X O"
		"- O O O X O X X"
		"- O O X O O X X"
		"- X O X X O X X"
		"- - O - X X X X"
		"- - O - - - - O"_pos, +0 / 2 // 26
	),
	Puzzle::Exact(
		"- - X O - O - -"
		"- - O O O O - -"
		"O O X O X X O -"
		"O O O O X X O O"
		"O O O X X O X -"
		"O X O X X X X X"
		"- - X X X X - -"
		"- - X - O - X -"_pos, -2 / 2 // 27
	),
	Puzzle::Exact(
		"- - O - - - - -"
		"- - O O O - - X"
		"- X O O O O X X"
		"X X X X O X O X"
		"- X X O X O O X"
		"X X O X O O X X"
		"- O O O O O - X"
		"- - - O O O - -"_pos, +0 / 2 // 28
	),
	Puzzle::Exact(
		"- O X X X X - -"
		"- - O X X O - -"
		"X X O O X O O O"
		"X X X O O X O O"
		"X X O O X O O O"
		"X X X X O O - X"
		"X - X X O - - -"
		"- - - - - - - -"_pos, +10 / 2 // 29
	),
	Puzzle::Exact(
		"- X X X - - - -"
		"X - X O O - - -"
		"X X O X O O - -"
		"X O X O X O - -"
		"X O O X O X X X"
		"X O O X X O X -"
		"- - O O O O O -"
		"- X X X X X - -"_pos, +0 / 2 // 30
	),
	Puzzle::Exact(
		"- O O O O O - -"
		"- - O O O O - -"
		"O X X O O O - -"
		"- X X X O O - -"
		"X X X X X X O -"
		"X X X O O O - O"
		"X - O O O O - -"
		"- O O O O O - -"_pos, -2 / 2 // 31
	),
	Puzzle::Exact(
		"- - X X - - - -"
		"O - X X O X - -"
		"O O X O O - - -"
		"O X O X O O O -"
		"O O X X O O O X"
		"O O X X X O O X"
		"- - X X X X O X"
		"- - X - - X - X"_pos, -4 / 2 // 32
	),
	Puzzle::Exact(
		"- X X X X X X X"
		"- - X O O O - -"
		"- - O X O O X X"
		"- O O X X O X X"
		"- O O O O O X X"
		"- X - X O O X X"
		"- - - O - X - X"
		"- - O O O O - -"_pos, -8 / 2 // 33
	),
	Puzzle::Exact(
		"- - - - - - - -"
		"- - - - - O - O"
		"- O O O O O O O"
		"O O O O O X O O"
		"O X X O O O X O"
		"- X X X O X O O"
		"- - X X X O X O"
		"- - O X X X X O"_pos, -2 / 2 // 34
	),
	Puzzle::Exact(
		"- - O O O - - -"
		"- - O O O O - X"
		"X X O O X X X X"
		"X X X X X X O X"
		"- X X O O O O X"
		"- X X X O O O X"
		"- - - O X O O -"
		"- - O - - - - -"_pos, +0 / 2 // 35
	),
	Puzzle::Exact(
		"- - - O - X - -"
		"- - O O O X - O"
		"O O O O O O O O"
		"O X X O O X X O"
		"O X O X X X O O"
		"O O X X X X - O"
		"O - - X X X X -"
		"- - - - - - - -"_pos, +0 / 2 // 36
	),
	Puzzle::Exact(
		"- - O O O O - -"
		"O - O O O O - -"
		"O X X X O O O -"
		"O X X O X O - -"
		"O O X X O X X -"
		"O O X X X X - -"
		"O - X X X - - -"
		"- - X X - O - -"_pos, -20 / 2 // 37
	),
	Puzzle::Exact(
		"- - O O O O - -"
		"- - O O O O - -"
		"- X O X X O O X"
		"O O X O O O O X"
		"- O O O O O X X"
		"X O O X X X X X"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +4 / 2 // 38
	),
	Puzzle::Exact(
		"X - X X X X - -"
		"O X O O X O - -"
		"O X X X O O O -"
		"O X X X O O - -"
		"O X X O X O - -"
		"O X O O O - - -"
		"O - O O - - - -"
		"- - - - - - - -"_pos, +64 / 2 // 39
	),
	Puzzle::Exact(
		"O - - O O O O X"
		"- O O O O O O X"
		"O O X X O O O X"
		"O O X O O O X X"
		"O O O O O O X X"
		"- - - O O O O X"
		"- - - - O - - X"
		"- - - - - - - -"_pos, +38 / 2 // 40
	),
	Puzzle::Exact(
		"- O O O O O - -"
		"- - O O O O X -"
		"- O O O O O O -"
		"X X X X X O O -"
		"- X X O O X - -"
		"O O X O X X - -"
		"- - O X X O - -"
		"- O O O - - O -"_pos, +0 / 2 // 41
	),
	Puzzle::Exact(
		"- - O O O - - -"
		"- - - - X X - O"
		"O O O O O X O O"
		"- O O O O X O O"
		"X - O O O X X O"
		"- - - O O X O O"
		"- - - O O O X O"
		"- - O O O O - -"_pos, +6 / 2 // 42
	),
	Puzzle::Exact(
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O O O O -"
		"- X X O O O X -"
		"X X X X O X X -"
		"- - - O X O - -"
		"- - O O O O O -"_pos, -12 / 2 // 43
	),
	Puzzle::Exact(
		"- - X - O - X -"
		"- - X - O X - X"
		"- X X O O O X X"
		"X X X X O O O X"
		"X X X X O O - -"
		"O O X X O X - -"
		"- - O O O O - -"
		"- - - O O O - -"_pos, -14 / 2 // 44
	),
	Puzzle::Exact(
		"- - - X X X X -"
		"X - X X X O - -"
		"X X O X O O - -"
		"X X X O X O - -"
		"X X O X X O - -"
		"- O X X X O O -"
		"O - O O O O - -"
		"- - - - O O - -"_pos, +6 / 2 // 45
	),
	Puzzle::Exact(
		"- - - X X X - -"
		"- - O O O X - -"
		"- - O O O X X -"
		"- O O O O X X X"
		"- - O O O O X X"
		"- - O X O X X X"
		"- - X X O O - -"
		"- X X X X - O -"_pos, -8 / 2 // 46
	),
	Puzzle::Exact(
		"- X X X X X - -"
		"- - X X X X - -"
		"- X X X X O - -"
		"O O O O O O - -"
		"- X O X X O - -"
		"X X X O X O - -"
		"- - X X O O - -"
		"- - O O O O - -"_pos, +4 / 2 // 47
	),
	Puzzle::Exact(
		"- - - - - O - -"
		"O - O O O - - -"
		"O O O O X X - -"
		"O X O X X O O -"
		"O X X O O O - -"
		"O X X O O - - -"
		"- - X X X O - -"
		"- O O O O O O -"_pos, +28 / 2 // 48
	),
	Puzzle::Exact(
		"- - O X - O - -"
		"- - X X O O - -"
		"O O O O O X X -"
		"O O O O O X - -"
		"O O O X O X X -"
		"O O O O X X - -"
		"- - - O O X - -"
		"- - X - O - - -"_pos, +16 / 2 // 49
	),
	Puzzle::Exact(
		"- - - - X - - -"
		"- - X X X - - -"
		"- O O O X O O O"
		"- O O O X O O O"
		"- O X O X O X O"
		"- O O X X O O O"
		"- - O O X O - -"
		"- - O - - O - -"_pos, +10 / 2 // 50
	),
	Puzzle::Exact(
		"- - - - X - O -"
		"- - - - - O - -"
		"- - - O O O X -"
		"X O O O O O X X"
		"- O O X X O X X"
		"O O X O O O X X"
		"- - X X X X - X"
		"- - - - X X - -"_pos, +6 / 2 // 51
	),
	Puzzle::Exact(
		"- - - O - - - -"
		"- - - X O - - O"
		"- - O X X O O O"
		"O O O X O O O O"
		"O O O X X O O O"
		"O O O X X X O O"
		"- - O X - - - O"
		"- - - - - - - -"_pos, +0 / 2 // 52
	),
	Puzzle::Exact(
		"- - - - O O - -"
		"- - - O O O - -"
		"- X X X X O O O"
		"- - X X O O X O"
		"- X X X X X O O"
		"- - O O O X O O"
		"- - X - O X - O"
		"- - - - - X - -"_pos, -2 / 2 // 53
	),
	Puzzle::Exact(
		"- - O O O - - -"
		"X X O O - - - -"
		"X X X X O O O O"
		"X X X X O X - -"
		"X X X O X X - -"
		"X X O O O - - -"
		"- - - O O O - -"
		"- - - O - - - -"_pos, -2 / 2 // 54
	),
	Puzzle::Exact(
		"- - - - - - - -"
		"O - O - - - - -"
		"- O O O O X X X"
		"X X O X O O - -"
		"X X X O O O O -"
		"X X O O O O - -"
		"X - X X X O - -"
		"- - - X X - - -"_pos, +0 / 2 // 55
	),
	Puzzle::Exact(
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O X O - -"
		"- X O O O O O -"
		"X X X X X O X -"
		"- - - X O O - -"
		"- - - - - - - -"_pos, +2 / 2 // 56
	),
	Puzzle::Exact(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X O O O"
		"- - X X X O O O"
		"- - X X O X O O"
		"- O O O X X X O"
		"- - O X O O - O"
		"- O O O O O - -"_pos, -10 / 2 // 57
	),
	Puzzle::Exact(
		"- - X O O O - -"
		"- - O O O - - -"
		"- O O O X O O -"
		"- O O O O X O -"
		"- O X O X X X -"
		"O O X X X X - -"
		"- - X - X X - -"
		"- - - - - - - -"_pos, +4 / 2 // 58
	),
	Puzzle::Exact(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - O"
		"- - O O O O O -"
		"- - O O O O O X"
		"O O O O X X X X"
		"- - X X O O X X"
		"- - X X - O - X"_pos, +64 / 2 // 59
	),
	Puzzle::Exact(
		"- - - O O O O -"
		"- - - O O O - -"
		"- - X O X O X X"
		"- - X O O X X X"
		"- - X O O X X X"
		"- - X O O O X X"
		"- - O X X X - X"
		"- - X X X X - -"_pos, +20 / 2 // 60
	),
	Puzzle::Exact(
		"- O O O O - - -"
		"O - O O X O - -"
		"O O O O X O O -"
		"O X X O X X X X"
		"O X X X X X X -"
		"O O X X X X - -"
		"O - - - X - - -"
		"- - - - - - - -"_pos, -14 / 2 // 61
	),
	Puzzle::Exact(
		"- - X X X X - -"
		"- - X X O O - -"
		"- - X O O O O O"
		"O O X O O X X X"
		"- O O O O X X -"
		"X O O O O O O X"
		"- - - - O - - -"
		"- - - - - - - -"_pos, +28 / 2 // 62
	),
	Puzzle::Exact(
		"- - O - - - - -"
		"- - O - O - - -"
		"- X O O O O - -"
		"- X O O O O X -"
		"X X O X O X X X"
		"X X X X X O X -"
		"- - O X O O - -"
		"- - - O O O O -"_pos, -2 / 2 // 63
	),
	Puzzle::Exact(
		"- - X - - O - -"
		"- - X - - O - X"
		"- X X O O O X X"
		"- - O O O O O X"
		"- - O O X X X X"
		"- O O O O O O -"
		"- - O O O - - -"
		"- O - X X X - -"_pos, +20 / 2 // 64
	),
	Puzzle::Exact(
		"- - - - O O - -"
		"- - O O O O X -"
		"- - O X X X X -"
		"O - O X X X X -"
		"- O O X X O X -"
		"X X O X X X X -"
		"- - O O O O - -"
		"- - - - - O - -"_pos, +10 / 2 // 65
	),
	Puzzle::Exact(
		"- O O O - - - -"
		"X - O X X - - -"
		"X X O X X O O -"
		"X O X X O O - -"
		"X X O O O O - -"
		"X X O O O O - -"
		"- - O O O - - -"
		"- - O - - - - -"_pos, +30 / 2 // 66
	),
	Puzzle::Exact(
		"- X X X X X - -"
		"- - X O X X - -"
		"O O O X O X O -"
		"- O O O X O O O"
		"- O O O X X O -"
		"- - O O O X - O"
		"- - - O X - - -"
		"- - - - - - - -"_pos, +22 / 2 // 67
	),
	Puzzle::Exact(
		"- - - O O O - -"
		"- - O O O O - -"
		"- - O X X O O X"
		"- O O X X O X -"
		"- O O X X X X -"
		"- X O O X X - -"
		"- - O O O - - -"
		"- - - - - O - -"_pos, +28 / 2 // 68
	),
	Puzzle::Exact(
		"- - O O O O - -"
		"- - - O O O - -"
		"- O O O O O - -"
		"X X O X X O O -"
		"- O X O X O O -"
		"O X X X X X X -"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +0 / 2 // 69
	),
	Puzzle::Exact(
		"- - - X - - - -"
		"X - X X X - - -"
		"X X X X - - - -"
		"X X X O O O - -"
		"X X X X O O - -"
		"X X O O X X X -"
		"X - O O X X - -"
		"- - O - - - - -"_pos, -24 / 2 // 70
	),
	Puzzle::Exact(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - O O O O O -"
		"- O O O O O X -"
		"- X O O O X X O"
		"- - X O X O O O"
		"- - X X O O - O"
		"- - - O O O O -"_pos, +20 / 2 // 71
	),
	Puzzle::Exact(
		"- - - X - - - -"
		"- - X X O O - -"
		"- O O X O O O -"
		"O O O O X X O O"
		"- O O O O X X -"
		"- - O O O X X -"
		"- - - O O - - -"
		"- - - - O - - -"_pos, +24 / 2 // 72
	),
	Puzzle::Exact(
		"- - O - - O - -"
		"- - O O O - - -"
		"X X O O O O - -"
		"- X X O O O - -"
		"- X O X O O X -"
		"X X X O O O O -"
		"X - - X O X - -"
		"- - - - - - - -"_pos, -4 / 2 // 73
	),
	Puzzle::Exact(
		"- - - - O - - -"
		"- - X O O X - O"
		"- - X O X X O O"
		"- X X O O X O O"
		"- - X O O X - O"
		"- - O O X X - -"
		"- - O X X X - -"
		"- - - - - X - -"_pos, -30 / 2 // 74
	),
	Puzzle::Exact(
		"- - - - O - - -"
		"- - - - O O - -"
		"- - X X O X - O"
		"- X X X O X O O"
		"- - O O O O O -"
		"- - O O O X O X"
		"- - O O O O X -"
		"- - - - - O - -"_pos, +14 / 2 // 75
	),
	Puzzle::Exact(
		"- - - O - - - -"
		"- - O O - O - -"
		"- - - O O O X -"
		"O O O O O O X -"
		"- X X X X O X X"
		"- - O O O O O O"
		"- - O O O - - -"
		"- - - - O - - -"_pos, +32 / 2 // 76
	),
	Puzzle::Exact(
		"- - O - O X - -"
		"X - O O O - - -"
		"X X O O O - - -"
		"X X O X O O O O"
		"- O O O O O - -"
		"O - X - O - - -"
		"- - O X - - - -"
		"- - - - - - - -"_pos, +34 / 2 // 77
	),
	Puzzle::Exact(
		"- - - - O - - -"
		"- - O O O O - -"
		"- O O O X - X -"
		"O O X O X X X X"
		"- X O O X - - -"
		"X O O O - X - -"
		"- - O O - - - -"
		"- - - O - - - -"_pos, +8 / 2 // 78
	),
	Puzzle::Exact(
		"- - - - - - - -"
		"- - - - - - X -"
		"- - - - O - X X"
		"- - - O O O X -"
		"O O O O X O X X"
		"- - O O O O O O"
		"- - O - O O - O"
		"- - - - O O - -"_pos, +64 / 2 // 79
	),
};

static Project FForum_1(FForum.begin() + 1, FForum.begin() + 20);
static Project FForum_2(FForum.begin() + 20, FForum.begin() + 40);
static Project FForum_3(FForum.begin() + 40, FForum.begin() + 60);
static Project FForum_4(FForum.begin() + 60, FForum.begin() + 80);
static ProjectDB FForums({FForum_1, FForum_2, FForum_3, FForum_4});