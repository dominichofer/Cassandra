#pragma once
#include "Search/Puzzle.h"
#include <vector>

static std::vector<Puzzle> FForum =
{
	Puzzle(Position::Start()),
	Puzzle::WithExactScore(
		"- - X X X X X -"
		"- O O O X X - O"
		"- O O O X X O X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- X X X O O O -"
		"- O O O O O - -"_pos, +18 / 2 // 01
	),
	Puzzle::WithExactScore(
		"- X X X X X X -"
		"- - X O O O O -"
		"- X O X X O O X"
		"X O O O O O O O"
		"O X O O X X O O"
		"O O O X X O O X"
		"- O O O O O - -"
		"- - X X X X X -"_pos, +10 / 2 // 02
	),
	Puzzle::WithExactScore(
		"- - - - O X - -"
		"- - O O X X - -"
		"- O O O X X - X"
		"O O X X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X X O X O"
		"- - O O O O O X"_pos, +2 / 2 // 03
	),
	Puzzle::WithExactScore(
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X X O X O O O X"
		"- O X O O X X X"
		"- - O O O X X X"
		"- - O O X X - -"
		"- - X O X X O -"_pos, +0 / 2 // 04
	),
	Puzzle::WithExactScore(
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O X O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O O - -"
		"- X X X X X - -"_pos, +32 / 2 // 05
	),
	Puzzle::WithExactScore(
		"- - O X X X - -"
		"O O O X X X - -"
		"O O O X O X O -"
		"O O X O O O X -"
		"O O X X X X X X"
		"X O O X X O X -"
		"- O O O O X - -"
		"- X X X X X X -"_pos, +14 / 2 // 06
	),
	Puzzle::WithExactScore(
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O X X X X"
		"X O O X X X X X"
		"X O O O O X X X"
		"- X X X X X X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 07
	),
	Puzzle::WithExactScore(
		"- - - O - O - -"
		"O - O O O O - -"
		"O O O O X O O O"
		"O O O X X X X X"
		"O O X O O O X -"
		"O X O O O O X -"
		"O X X O O O - -"
		"O X X O O X - -"_pos, +8 / 2 // 08
	),
	Puzzle::WithExactScore(
		"- - O X O O - -"
		"X - X X O O O O"
		"- X X X O O O O"
		"- O X O O O X O"
		"O O X O X X X O"
		"X O O X O X O O"
		"- - X O X X - -"
		"- - X X X X - -"_pos, -8 / 2 // 09
	),
	Puzzle::WithExactScore(
		"- O O O O - - -"
		"- - X O O O - -"
		"O X O X O X O O"
		"X O X O O X O O"
		"X O O X O X X X"
		"O O O X O X X O"
		"- - X O O X - -"
		"- X X X X X - -"_pos, +10 / 2 // 10
	),
	Puzzle::WithExactScore(
		"- - - X - O X O"
		"- - - - O O X O"
		"- - - O O X X O"
		"X - O O X O X O"
		"O O O X X O X O"
		"- O X X O O O O"
		"O X X X O O - O"
		"X X X X X X X -"_pos, +30 / 2 // 11
	),
	Puzzle::WithExactScore(
		"- - X - - X - -"
		"O - X X X X O -"
		"O O X X X O X X"
		"O O X O X O X X"
		"O O X O O X X X"
		"O O O O X X X X"
		"- - X O O O - -"
		"- O O O O O - -"_pos, -8 / 2 // 12
	),
	Puzzle::WithExactScore(
		"- - X X X X X -"
		"- O O O X X - -"
		"- O O O X X X X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- - O X O O O -"
		"- O O O O O - -"_pos, +14 / 2 // 13
	),
	Puzzle::WithExactScore(
		"- - X X X X X -"
		"- - O O O X - -"
		"- X O O X X X X"
		"- O O O O O O O"
		"O O O X X X O O"
		"O O O X X O O X"
		"- - X X O O - -"
		"- - X X X X X -"_pos, +18 / 2 // 14
	),
	Puzzle::WithExactScore(
		"- - - - O - - -"
		"- - - O O X - -"
		"- O O O X X - X"
		"O O O X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X O O X O"
		"- - O O O O O X"_pos, +4 / 2 // 15
	),
	Puzzle::WithExactScore(
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X O O X X X O X"
		"- O O O X X X X"
		"- - O O X X X X"
		"- - - O O O - -"
		"- - X O X - O -"_pos, +24 / 2 // 16
	),
	Puzzle::WithExactScore(
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O O O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O - - -"
		"- X X X X - - -"_pos, +8 / 2 // 17
	),
	Puzzle::WithExactScore(
		"- X X X - - - -"
		"- - O O O X - -"
		"X O O O O O X X"
		"O X O X O O X X"
		"O X X O O O O O"
		"X X X O X O O X"
		"- - O X X O - -"
		"- O O O O O - -"_pos, -2 / 2 // 18
	),
	Puzzle::WithExactScore(
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O O X X X"
		"X O O O X X X X"
		"X - O O O X X X"
		"- - O O O O X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 19
	),
	Puzzle::WithExactScore(
		"X X X O X X X X"
		"O X X X X X X X"
		"O O X X X X X X"
		"O O O X X X X X"
		"O O O X X O O -"
		"O O O O O - - -"
		"O O O O O O O -"
		"O O O O O O O -"_pos, +6 / 2 // 20
	),
	Puzzle::WithExactScore(
		"X X X X X X X X"
		"O X X O O O - -"
		"O O X X O X X -"
		"O X O X X X - -"
		"O X X X X O - -"
		"O X X O X X - -"
		"O X X X X X - -"
		"O O O O - - - -"_pos, +0 / 2 // 21
	),
	Puzzle::WithExactScore(
		"- - X X X X - -"
		"O - X X X X X -"
		"O O X X O X O O"
		"O X O X O O O O"
		"O O O X O O O O"
		"- O O X O X O O"
		"- - X O O O - O"
		"- - - - O - - -"_pos, +2 / 2 // 22
	),
	Puzzle::WithExactScore(
		"- - O - - - - -"
		"- - O O X - - -"
		"O O O X X X O -"
		"O O O O X O X X"
		"X X X O O X O X"
		"X X X X X O O X"
		"X - X X X X O X"
		"- - X X X X - -"_pos, +4 / 2 // 23
	),
	Puzzle::WithExactScore(
		"- - X - - X - -"
		"- - - X X X O -"
		"- O - O X O X X"
		"- - O O O X X X"
		"O O O O X X X X"
		"O O O X O O X X"
		"O O O O O O - -"
		"O X O O - X - -"_pos, +0 / 2 // 24
	),
	Puzzle::WithExactScore(
		"- - - - O - - -"
		"- - - O O O X -"
		"- X X X O O O O"
		"O X X X X O O X"
		"- O O X X O O X"
		"X X O X O O O O"
		"X X X O O - - -"
		"O - O O O O - -"_pos, +0 / 2 // 25
	),
	Puzzle::WithExactScore(
		"- O O O O O - -"
		"- - O X X O - -"
		"- O O O O X X O"
		"- O O O X O X X"
		"- O O X O O X X"
		"- X O X X O X X"
		"- - O - X X X X"
		"- - O - - - - O"_pos, +0 / 2 // 26
	),
	Puzzle::WithExactScore(
		"- - X O - O - -"
		"- - O O O O - -"
		"O O X O X X O -"
		"O O O O X X O O"
		"O O O X X O X -"
		"O X O X X X X X"
		"- - X X X X - -"
		"- - X - O - X -"_pos, -2 / 2 // 27
	),
	Puzzle::WithExactScore(
		"- - O - - - - -"
		"- - O O O - - X"
		"- X O O O O X X"
		"X X X X O X O X"
		"- X X O X O O X"
		"X X O X O O X X"
		"- O O O O O - X"
		"- - - O O O - -"_pos, +0 / 2 // 28
	),
	Puzzle::WithExactScore(
		"- O X X X X - -"
		"- - O X X O - -"
		"X X O O X O O O"
		"X X X O O X O O"
		"X X O O X O O O"
		"X X X X O O - X"
		"X - X X O - - -"
		"- - - - - - - -"_pos, +10 / 2 // 29
	),
	Puzzle::WithExactScore(
		"- X X X - - - -"
		"X - X O O - - -"
		"X X O X O O - -"
		"X O X O X O - -"
		"X O O X O X X X"
		"X O O X X O X -"
		"- - O O O O O -"
		"- X X X X X - -"_pos, +0 / 2 // 30
	),
	Puzzle::WithExactScore(
		"- O O O O O - -"
		"- - O O O O - -"
		"O X X O O O - -"
		"- X X X O O - -"
		"X X X X X X O -"
		"X X X O O O - O"
		"X - O O O O - -"
		"- O O O O O - -"_pos, -2 / 2 // 31
	),
	Puzzle::WithExactScore(
		"- - X X - - - -"
		"O - X X O X - -"
		"O O X O O - - -"
		"O X O X O O O -"
		"O O X X O O O X"
		"O O X X X O O X"
		"- - X X X X O X"
		"- - X - - X - X"_pos, -4 / 2 // 32
	),
	Puzzle::WithExactScore(
		"- X X X X X X X"
		"- - X O O O - -"
		"- - O X O O X X"
		"- O O X X O X X"
		"- O O O O O X X"
		"- X - X O O X X"
		"- - - O - X - X"
		"- - O O O O - -"_pos, -8 / 2 // 33
	),
	Puzzle::WithExactScore(
		"- - - - - - - -"
		"- - - - - O - O"
		"- O O O O O O O"
		"O O O O O X O O"
		"O X X O O O X O"
		"- X X X O X O O"
		"- - X X X O X O"
		"- - O X X X X O"_pos, -2 / 2 // 34
	),
	Puzzle::WithExactScore(
		"- - O O O - - -"
		"- - O O O O - X"
		"X X O O X X X X"
		"X X X X X X O X"
		"- X X O O O O X"
		"- X X X O O O X"
		"- - - O X O O -"
		"- - O - - - - -"_pos, +0 / 2 // 35
	),
	Puzzle::WithExactScore(
		"- - - O - X - -"
		"- - O O O X - O"
		"O O O O O O O O"
		"O X X O O X X O"
		"O X O X X X O O"
		"O O X X X X - O"
		"O - - X X X X -"
		"- - - - - - - -"_pos, +0 / 2 // 36
	),
	Puzzle::WithExactScore(
		"- - O O O O - -"
		"O - O O O O - -"
		"O X X X O O O -"
		"O X X O X O - -"
		"O O X X O X X -"
		"O O X X X X - -"
		"O - X X X - - -"
		"- - X X - O - -"_pos, -20 / 2 // 37
	),
	Puzzle::WithExactScore(
		"- - O O O O - -"
		"- - O O O O - -"
		"- X O X X O O X"
		"O O X O O O O X"
		"- O O O O O X X"
		"X O O X X X X X"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +4 / 2 // 38
	),
	Puzzle::WithExactScore(
		"X - X X X X - -"
		"O X O O X O - -"
		"O X X X O O O -"
		"O X X X O O - -"
		"O X X O X O - -"
		"O X O O O - - -"
		"O - O O - - - -"
		"- - - - - - - -"_pos, +64 / 2 // 39
	),
	Puzzle::WithExactScore(
		"O - - O O O O X"
		"- O O O O O O X"
		"O O X X O O O X"
		"O O X O O O X X"
		"O O O O O O X X"
		"- - - O O O O X"
		"- - - - O - - X"
		"- - - - - - - -"_pos, +38 / 2 // 40
	),
	Puzzle::WithExactScore(
		"- O O O O O - -"
		"- - O O O O X -"
		"- O O O O O O -"
		"X X X X X O O -"
		"- X X O O X - -"
		"O O X O X X - -"
		"- - O X X O - -"
		"- O O O - - O -"_pos, +0 / 2 // 41
	),
	Puzzle::WithExactScore(
		"- - O O O - - -"
		"- - - - X X - O"
		"O O O O O X O O"
		"- O O O O X O O"
		"X - O O O X X O"
		"- - - O O X O O"
		"- - - O O O X O"
		"- - O O O O - -"_pos, +6 / 2 // 42
	),
	Puzzle::WithExactScore(
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O O O O -"
		"- X X O O O X -"
		"X X X X O X X -"
		"- - - O X O - -"
		"- - O O O O O -"_pos, -12 / 2 // 43
	),
	Puzzle::WithExactScore(
		"- - X - O - X -"
		"- - X - O X - X"
		"- X X O O O X X"
		"X X X X O O O X"
		"X X X X O O - -"
		"O O X X O X - -"
		"- - O O O O - -"
		"- - - O O O - -"_pos, -14 / 2 // 44
	),
	Puzzle::WithExactScore(
		"- - - X X X X -"
		"X - X X X O - -"
		"X X O X O O - -"
		"X X X O X O - -"
		"X X O X X O - -"
		"- O X X X O O -"
		"O - O O O O - -"
		"- - - - O O - -"_pos, +6 / 2 // 45
	),
	Puzzle::WithExactScore(
		"- - - X X X - -"
		"- - O O O X - -"
		"- - O O O X X -"
		"- O O O O X X X"
		"- - O O O O X X"
		"- - O X O X X X"
		"- - X X O O - -"
		"- X X X X - O -"_pos, -8 / 2 // 46
	),
	Puzzle::WithExactScore(
		"- X X X X X - -"
		"- - X X X X - -"
		"- X X X X O - -"
		"O O O O O O - -"
		"- X O X X O - -"
		"X X X O X O - -"
		"- - X X O O - -"
		"- - O O O O - -"_pos, +4 / 2 // 47
	),
	Puzzle::WithExactScore(
		"- - - - - O - -"
		"O - O O O - - -"
		"O O O O X X - -"
		"O X O X X O O -"
		"O X X O O O - -"
		"O X X O O - - -"
		"- - X X X O - -"
		"- O O O O O O -"_pos, +28 / 2 // 48
	),
	Puzzle::WithExactScore(
		"- - O X - O - -"
		"- - X X O O - -"
		"O O O O O X X -"
		"O O O O O X - -"
		"O O O X O X X -"
		"O O O O X X - -"
		"- - - O O X - -"
		"- - X - O - - -"_pos, +16 / 2 // 49
	),
	Puzzle::WithExactScore(
		"- - - - X - - -"
		"- - X X X - - -"
		"- O O O X O O O"
		"- O O O X O O O"
		"- O X O X O X O"
		"- O O X X O O O"
		"- - O O X O - -"
		"- - O - - O - -"_pos, +10 / 2 // 50
	),
	Puzzle::WithExactScore(
		"- - - - X - O -"
		"- - - - - O - -"
		"- - - O O O X -"
		"X O O O O O X X"
		"- O O X X O X X"
		"O O X O O O X X"
		"- - X X X X - X"
		"- - - - X X - -"_pos, +6 / 2 // 51
	),
	Puzzle::WithExactScore(
		"- - - O - - - -"
		"- - - X O - - O"
		"- - O X X O O O"
		"O O O X O O O O"
		"O O O X X O O O"
		"O O O X X X O O"
		"- - O X - - - O"
		"- - - - - - - -"_pos, +0 / 2 // 52
	),
	Puzzle::WithExactScore(
		"- - - - O O - -"
		"- - - O O O - -"
		"- X X X X O O O"
		"- - X X O O X O"
		"- X X X X X O O"
		"- - O O O X O O"
		"- - X - O X - O"
		"- - - - - X - -"_pos, -2 / 2 // 53
	),
	Puzzle::WithExactScore(
		"- - O O O - - -"
		"X X O O - - - -"
		"X X X X O O O O"
		"X X X X O X - -"
		"X X X O X X - -"
		"X X O O O - - -"
		"- - - O O O - -"
		"- - - O - - - -"_pos, -2 / 2 // 54
	),
	Puzzle::WithExactScore(
		"- - - - - - - -"
		"O - O - - - - -"
		"- O O O O X X X"
		"X X O X O O - -"
		"X X X O O O O -"
		"X X O O O O - -"
		"X - X X X O - -"
		"- - - X X - - -"_pos, +0 / 2 // 55
	),
	Puzzle::WithExactScore(
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O X O - -"
		"- X O O O O O -"
		"X X X X X O X -"
		"- - - X O O - -"
		"- - - - - - - -"_pos, +2 / 2 // 56
	),
	Puzzle::WithExactScore(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X O O O"
		"- - X X X O O O"
		"- - X X O X O O"
		"- O O O X X X O"
		"- - O X O O - O"
		"- O O O O O - -"_pos, -10 / 2 // 57
	),
	Puzzle::WithExactScore(
		"- - X O O O - -"
		"- - O O O - - -"
		"- O O O X O O -"
		"- O O O O X O -"
		"- O X O X X X -"
		"O O X X X X - -"
		"- - X - X X - -"
		"- - - - - - - -"_pos, +4 / 2 // 58
	),
	Puzzle::WithExactScore(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - O"
		"- - O O O O O -"
		"- - O O O O O X"
		"O O O O X X X X"
		"- - X X O O X X"
		"- - X X - O - X"_pos, +64 / 2 // 59
	),
	Puzzle::WithExactScore(
		"- - - O O O O -"
		"- - - O O O - -"
		"- - X O X O X X"
		"- - X O O X X X"
		"- - X O O X X X"
		"- - X O O O X X"
		"- - O X X X - X"
		"- - X X X X - -"_pos, +20 / 2 // 60
	),
	Puzzle::WithExactScore(
		"- O O O O - - -"
		"O - O O X O - -"
		"O O O O X O O -"
		"O X X O X X X X"
		"O X X X X X X -"
		"O O X X X X - -"
		"O - - - X - - -"
		"- - - - - - - -"_pos, -14 / 2 // 61
	),
	Puzzle::WithExactScore(
		"- - X X X X - -"
		"- - X X O O - -"
		"- - X O O O O O"
		"O O X O O X X X"
		"- O O O O X X -"
		"X O O O O O O X"
		"- - - - O - - -"
		"- - - - - - - -"_pos, +28 / 2 // 62
	),
	Puzzle::WithExactScore(
		"- - O - - - - -"
		"- - O - O - - -"
		"- X O O O O - -"
		"- X O O O O X -"
		"X X O X O X X X"
		"X X X X X O X -"
		"- - O X O O - -"
		"- - - O O O O -"_pos, -2 / 2 // 63
	),
	Puzzle::WithExactScore(
		"- - X - - O - -"
		"- - X - - O - X"
		"- X X O O O X X"
		"- - O O O O O X"
		"- - O O X X X X"
		"- O O O O O O -"
		"- - O O O - - -"
		"- O - X X X - -"_pos, +20 / 2 // 64
	),
	Puzzle::WithExactScore(
		"- - - - O O - -"
		"- - O O O O X -"
		"- - O X X X X -"
		"O - O X X X X -"
		"- O O X X O X -"
		"X X O X X X X -"
		"- - O O O O - -"
		"- - - - - O - -"_pos, +10 / 2 // 65
	),
	Puzzle::WithExactScore(
		"- O O O - - - -"
		"X - O X X - - -"
		"X X O X X O O -"
		"X O X X O O - -"
		"X X O O O O - -"
		"X X O O O O - -"
		"- - O O O - - -"
		"- - O - - - - -"_pos, +30 / 2 // 66
	),
	Puzzle::WithExactScore(
		"- X X X X X - -"
		"- - X O X X - -"
		"O O O X O X O -"
		"- O O O X O O O"
		"- O O O X X O -"
		"- - O O O X - O"
		"- - - O X - - -"
		"- - - - - - - -"_pos, +22 / 2 // 67
	),
	Puzzle::WithExactScore(
		"- - - O O O - -"
		"- - O O O O - -"
		"- - O X X O O X"
		"- O O X X O X -"
		"- O O X X X X -"
		"- X O O X X - -"
		"- - O O O - - -"
		"- - - - - O - -"_pos, +28 / 2 // 68
	),
	Puzzle::WithExactScore(
		"- - O O O O - -"
		"- - - O O O - -"
		"- O O O O O - -"
		"X X O X X O O -"
		"- O X O X O O -"
		"O X X X X X X -"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +0 / 2 // 69
	),
	Puzzle::WithExactScore(
		"- - - X - - - -"
		"X - X X X - - -"
		"X X X X - - - -"
		"X X X O O O - -"
		"X X X X O O - -"
		"X X O O X X X -"
		"X - O O X X - -"
		"- - O - - - - -"_pos, -24 / 2 // 70
	),
	Puzzle::WithExactScore(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - O O O O O -"
		"- O O O O O X -"
		"- X O O O X X O"
		"- - X O X O O O"
		"- - X X O O - O"
		"- - - O O O O -"_pos, +20 / 2 // 71
	),
	Puzzle::WithExactScore(
		"- - - X - - - -"
		"- - X X O O - -"
		"- O O X O O O -"
		"O O O O X X O O"
		"- O O O O X X -"
		"- - O O O X X -"
		"- - - O O - - -"
		"- - - - O - - -"_pos, +24 / 2 // 72
	),
	Puzzle::WithExactScore(
		"- - O - - O - -"
		"- - O O O - - -"
		"X X O O O O - -"
		"- X X O O O - -"
		"- X O X O O X -"
		"X X X O O O O -"
		"X - - X O X - -"
		"- - - - - - - -"_pos, -4 / 2 // 73
	),
	Puzzle::WithExactScore(
		"- - - - O - - -"
		"- - X O O X - O"
		"- - X O X X O O"
		"- X X O O X O O"
		"- - X O O X - O"
		"- - O O X X - -"
		"- - O X X X - -"
		"- - - - - X - -"_pos, -30 / 2 // 74
	),
	Puzzle::WithExactScore(
		"- - - - O - - -"
		"- - - - O O - -"
		"- - X X O X - O"
		"- X X X O X O O"
		"- - O O O O O -"
		"- - O O O X O X"
		"- - O O O O X -"
		"- - - - - O - -"_pos, +14 / 2 // 75
	),
	Puzzle::WithExactScore(
		"- - - O - - - -"
		"- - O O - O - -"
		"- - - O O O X -"
		"O O O O O O X -"
		"- X X X X O X X"
		"- - O O O O O O"
		"- - O O O - - -"
		"- - - - O - - -"_pos, +32 / 2 // 76
	),
	Puzzle::WithExactScore(
		"- - O - O X - -"
		"X - O O O - - -"
		"X X O O O - - -"
		"X X O X O O O O"
		"- O O O O O - -"
		"O - X - O - - -"
		"- - O X - - - -"
		"- - - - - - - -"_pos, +34 / 2 // 77
	),
	Puzzle::WithExactScore(
		"- - - - O - - -"
		"- - O O O O - -"
		"- O O O X - X -"
		"O O X O X X X X"
		"- X O O X - - -"
		"X O O O - X - -"
		"- - O O - - - -"
		"- - - O - - - -"_pos, +8 / 2 // 78
	),
	Puzzle::WithExactScore(
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

static std::vector FForum_1(FForum.begin() + 1, FForum.begin() + 20);
static std::vector FForum_2(FForum.begin() + 20, FForum.begin() + 40);
static std::vector FForum_3(FForum.begin() + 40, FForum.begin() + 60);
static std::vector FForum_4(FForum.begin() + 60, FForum.begin() + 80);
static std::vector FForums{ FForum_1, FForum_2, FForum_3, FForum_4 };