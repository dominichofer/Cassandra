#pragma once
#include "Core/Core.h"
#include <vector>

struct PosScore
{
	Position pos;
	int score;
};

inline int EmptyCount(const PosScore& ps) { return ps.pos.EmptyCount(); }

static std::vector<PosScore> FForum =
{
	PosScore(Position::Start()),
	PosScore(
		"- - X X X X X -"
		"- O O O X X - O"
		"- O O O X X O X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- X X X O O O -"
		"- O O O O O - -"_pos, +18 / 2 // 01
	),
	PosScore(
		"- X X X X X X -"
		"- - X O O O O -"
		"- X O X X O O X"
		"- O O O O O O O"
		"O O O O X X O O"
		"O O O X X O O X"
		"- - X X O O - -"
		"- - X X X X X -"_pos, +10 / 2 // 02
	),
	PosScore(
		"- - - - O X - -"
		"- - O O X X - -"
		"- O O O X X - X"
		"O O X X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X X O X O"
		"- - O O O O O X"_pos, +2 / 2 // 03
	),
	PosScore(
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X X O X O O O X"
		"- O X O O X X X"
		"- - O O O X X X"
		"- - O O X X - -"
		"- - X O X X O -"_pos, +0 / 2 // 04
	),
	PosScore(
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O X O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O O - -"
		"- X X X X X - -"_pos, +32 / 2 // 05
	),
	PosScore(
		"- - O X X X - -"
		"O O O X X X - -"
		"O O O X O X O -"
		"O O X O O O X -"
		"O O X X X X X X"
		"X O O X X O X -"
		"- O O O O X - -"
		"- X X X X X X -"_pos, +14 / 2 // 06
	),
	PosScore(
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O X X X X"
		"X O O X X X X X"
		"X O O O O X X X"
		"- X X X X X X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 07
	),
	PosScore(
		"- - - O - O - -"
		"O - O O O O - -"
		"O O O O X O O O"
		"O O O X X X X X"
		"O O X O O O X -"
		"O X O O O O X -"
		"O X X O O O - -"
		"O X X O O X - -"_pos, +8 / 2 // 08
	),
	PosScore(
		"- - O X O O - -"
		"X - X X O O O O"
		"- X X X O O O O"
		"- O X O O O X O"
		"O O X O X X X O"
		"X O O X O X O O"
		"- - X O X X - -"
		"- - X X X X - -"_pos, -8 / 2 // 09
	),
	PosScore(
		"- O O O O - - -"
		"- - X O O O - -"
		"O X O X O X O O"
		"X O X O O X O O"
		"X O O X O X X X"
		"O O O X O X X O"
		"- - X O O X - -"
		"- X X X X X - -"_pos, +10 / 2 // 10
	),
	PosScore(
		"- - - X - O X O"
		"- - - - O O X O"
		"- - - O O X X O"
		"X - O O X O X O"
		"O O O X X O X O"
		"- O X X O O O O"
		"O X X X O O - O"
		"X X X X X X X -"_pos, +30 / 2 // 11
	),
	PosScore(
		"- - X - - X - -"
		"O - X X X X O -"
		"O O X X X O X X"
		"O O X O X O X X"
		"O O X O O X X X"
		"O O O O X X X X"
		"- - X O O O - -"
		"- O O O O O - -"_pos, -8 / 2 // 12
	),
	PosScore(
		"- - X X X X X -"
		"- O O O X X - -"
		"- O O O X X X X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- - O X O O O -"
		"- O O O O O - -"_pos, +14 / 2 // 13
	),
	PosScore(
		"- - X X X X X -"
		"- - O O O X - -"
		"- X O O X X X X"
		"- O O O O O O O"
		"O O O X X X O O"
		"O O O X X O O X"
		"- - X X O O - -"
		"- - X X X X X -"_pos, +18 / 2 // 14
	),
	PosScore(
		"- - - - O - - -"
		"- - - O O X - -"
		"- O O O X X - X"
		"O O O X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X O O X O"
		"- - O O O O O X"_pos, +4 / 2 // 15
	),
	PosScore(
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X O O X X X O X"
		"- O O O X X X X"
		"- - O O X X X X"
		"- - - O O O - -"
		"- - X O X - O -"_pos, +24 / 2 // 16
	),
	PosScore(
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O O O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O - - -"
		"- X X X X - - -"_pos, +8 / 2 // 17
	),
	PosScore(
		"- X X X - - - -"
		"- - O O O X - -"
		"X O O O O O X X"
		"O X O X O O X X"
		"O X X O O O O O"
		"X X X O X O O X"
		"- - O X X O - -"
		"- O O O O O - -"_pos, -2 / 2 // 18
	),
	PosScore(
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O O X X X"
		"X O O O X X X X"
		"X - O O O X X X"
		"- - O O O O X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 19
	),
	PosScore(
		"X X X O X X X X"
		"O X X X X X X X"
		"O O X X X X X X"
		"O O O X X X X X"
		"O O O X X O O -"
		"O O O O O - - -"
		"O O O O O O O -"
		"O O O O O O O -"_pos, +6 / 2 // 20
	),
	PosScore(
		"X X X X X X X X"
		"O X X O O O - -"
		"O O X X O X X -"
		"O X O X X X - -"
		"O X X X X O - -"
		"O X X O X X - -"
		"O X X X X X - -"
		"O O O O - - - -"_pos, +0 / 2 // 21
	),
	PosScore(
		"- - X X X X - -"
		"O - X X X X X -"
		"O O X X O X O O"
		"O X O X O O O O"
		"O O O X O O O O"
		"- O O X O X O O"
		"- - X O O O - O"
		"- - - - O - - -"_pos, +2 / 2 // 22
	),
	PosScore(
		"- - O - - - - -"
		"- - O O X - - -"
		"O O O X X X O -"
		"O O O O X O X X"
		"X X X O O X O X"
		"X X X X X O O X"
		"X - X X X X O X"
		"- - X X X X - -"_pos, +4 / 2 // 23
	),
	PosScore(
		"- - X - - X - -"
		"- - - X X X O -"
		"- O - O X O X X"
		"- - O O O X X X"
		"O O O O X X X X"
		"O O O X O O X X"
		"O O O O O O - -"
		"O X O O - X - -"_pos, +0 / 2 // 24
	),
	PosScore(
		"- - - - O - - -"
		"- - - O O O X -"
		"- X X X O O O O"
		"O X X X X O O X"
		"- O O X X O O X"
		"X X O X O O O O"
		"X X X O O - - -"
		"O - O O O O - -"_pos, +0 / 2 // 25
	),
	PosScore(
		"- O O O O O - -"
		"- - O X X O - -"
		"- O O O O X X O"
		"- O O O X O X X"
		"- O O X O O X X"
		"- X O X X O X X"
		"- - O - X X X X"
		"- - O - - - - O"_pos, +0 / 2 // 26
	),
	PosScore(
		"- - X O - O - -"
		"- - O O O O - -"
		"O O X O X X O -"
		"O O O O X X O O"
		"O O O X X O X -"
		"O X O X X X X X"
		"- - X X X X - -"
		"- - X - O - X -"_pos, -2 / 2 // 27
	),
	PosScore(
		"- - O - - - - -"
		"- - O O O - - X"
		"- X O O O O X X"
		"X X X X O X O X"
		"- X X O X O O X"
		"X X O X O O X X"
		"- O O O O O - X"
		"- - - O O O - -"_pos, +0 / 2 // 28
	),
	PosScore(
		"- O X X X X - -"
		"- - O X X O - -"
		"X X O O X O O O"
		"X X X O O X O O"
		"X X O O X O O O"
		"X X X X O O - X"
		"X - X X O - - -"
		"- - - - - - - -"_pos, +10 / 2 // 29
	),
	PosScore(
		"- X X X - - - -"
		"X - X O O - - -"
		"X X O X O O - -"
		"X O X O X O - -"
		"X O O X O X X X"
		"X O O X X O X -"
		"- - O O O O O -"
		"- X X X X X - -"_pos, +0 / 2 // 30
	),
	PosScore(
		"- O O O O O - -"
		"- - O O O O - -"
		"O X X O O O - -"
		"- X X X O O - -"
		"X X X X X X O -"
		"X X X O O O - O"
		"X - O O O O - -"
		"- O O O O O - -"_pos, -2 / 2 // 31
	),
	PosScore(
		"- - X X - - - -"
		"O - X X O X - -"
		"O O X O O - - -"
		"O X O X O O O -"
		"O O X X O O O X"
		"O O X X X O O X"
		"- - X X X X O X"
		"- - X - - X - X"_pos, -4 / 2 // 32
	),
	PosScore(
		"- X X X X X X X"
		"- - X O O O - -"
		"- - O X O O X X"
		"- O O X X O X X"
		"- O O O O O X X"
		"- X - X O O X X"
		"- - - O - X - X"
		"- - O O O O - -"_pos, -8 / 2 // 33
	),
	PosScore(
		"- - - - - - - -"
		"- - - - - O - O"
		"- O O O O O O O"
		"O O O O O X O O"
		"O X X O O O X O"
		"- X X X O X O O"
		"- - X X X O X O"
		"- - O X X X X O"_pos, -2 / 2 // 34
	),
	PosScore(
		"- - O O O - - -"
		"- - O O O O - X"
		"X X O O X X X X"
		"X X X X X X O X"
		"- X X O O O O X"
		"- X X X O O O X"
		"- - - O X O O -"
		"- - O - - - - -"_pos, +0 / 2 // 35
	),
	PosScore(
		"- - - O - X - -"
		"- - O O O X - O"
		"O O O O O O O O"
		"O X X O O X X O"
		"O X O X X X O O"
		"O O X X X X - O"
		"O - - X X X X -"
		"- - - - - - - -"_pos, +0 / 2 // 36
	),
	PosScore(
		"- - O O O O - -"
		"O - O O O O - -"
		"O X X X O O O -"
		"O X X O X O - -"
		"O O X X O X X -"
		"O O X X X X - -"
		"O - X X X - - -"
		"- - X X - O - -"_pos, -20 / 2 // 37
	),
	PosScore(
		"- - O O O O - -"
		"- - O O O O - -"
		"- X O X X O O X"
		"O O X O O O O X"
		"- O O O O O X X"
		"X O O X X X X X"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +4 / 2 // 38
	),
	PosScore(
		"X - X X X X - -"
		"O X O O X O - -"
		"O X X X O O O -"
		"O X X X O O - -"
		"O X X O X O - -"
		"O X O O O - - -"
		"O - O O - - - -"
		"- - - - - - - -"_pos, +64 / 2 // 39
	),
	PosScore(
		"O - - O O O O X"
		"- O O O O O O X"
		"O O X X O O O X"
		"O O X O O O X X"
		"O O O O O O X X"
		"- - - O O O O X"
		"- - - - O - - X"
		"- - - - - - - -"_pos, +38 / 2 // 40
	),
	PosScore(
		"- O O O O O - -"
		"- - O O O O X -"
		"- O O O O O O -"
		"X X X X X O O -"
		"- X X O O X - -"
		"O O X O X X - -"
		"- - O X X O - -"
		"- O O O - - O -"_pos, +0 / 2 // 41
	),
	PosScore(
		"- - O O O - - -"
		"- - - - X X - O"
		"O O O O O X O O"
		"- O O O O X O O"
		"X - O O O X X O"
		"- - - O O X O O"
		"- - - O O O X O"
		"- - O O O O - -"_pos, +6 / 2 // 42
	),
	PosScore(
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O O O O -"
		"- X X O O O X -"
		"X X X X O X X -"
		"- - - O X O - -"
		"- - O O O O O -"_pos, -12 / 2 // 43
	),
	PosScore(
		"- - X - O - X -"
		"- - X - O X - X"
		"- X X O O O X X"
		"X X X X O O O X"
		"X X X X O O - -"
		"O O X X O X - -"
		"- - O O O O - -"
		"- - - O O O - -"_pos, -14 / 2 // 44
	),
	PosScore(
		"- - - X X X X -"
		"X - X X X O - -"
		"X X O X O O - -"
		"X X X O X O - -"
		"X X O X X O - -"
		"- O X X X O O -"
		"O - O O O O - -"
		"- - - - O O - -"_pos, +6 / 2 // 45
	),
	PosScore(
		"- - - X X X - -"
		"- - O O O X - -"
		"- - O O O X X -"
		"- O O O O X X X"
		"- - O O O O X X"
		"- - O X O X X X"
		"- - X X O O - -"
		"- X X X X - O -"_pos, -8 / 2 // 46
	),
	PosScore(
		"- X X X X X - -"
		"- - X X X X - -"
		"- X X X X O - -"
		"O O O O O O - -"
		"- X O X X O - -"
		"X X X O X O - -"
		"- - X X O O - -"
		"- - O O O O - -"_pos, +4 / 2 // 47
	),
	PosScore(
		"- - - - - O - -"
		"O - O O O - - -"
		"O O O O X X - -"
		"O X O X X O O -"
		"O X X O O O - -"
		"O X X O O - - -"
		"- - X X X O - -"
		"- O O O O O O -"_pos, +28 / 2 // 48
	),
	PosScore(
		"- - O X - O - -"
		"- - X X O O - -"
		"O O O O O X X -"
		"O O O O O X - -"
		"O O O X O X X -"
		"O O O O X X - -"
		"- - - O O X - -"
		"- - X - O - - -"_pos, +16 / 2 // 49
	),
	PosScore(
		"- - - - X - - -"
		"- - X X X - - -"
		"- O O O X O O O"
		"- O O O X O O O"
		"- O X O X O X O"
		"- O O X X O O O"
		"- - O O X O - -"
		"- - O - - O - -"_pos, +10 / 2 // 50
	),
	PosScore(
		"- - - - X - O -"
		"- - - - - O - -"
		"- - - O O O X -"
		"X O O O O O X X"
		"- O O X X O X X"
		"O O X O O O X X"
		"- - X X X X - X"
		"- - - - X X - -"_pos, +6 / 2 // 51
	),
	PosScore(
		"- - - O - - - -"
		"- - - X O - - O"
		"- - O X X O O O"
		"O O O X O O O O"
		"O O O X X O O O"
		"O O O X X X O O"
		"- - O X - - - O"
		"- - - - - - - -"_pos, +0 / 2 // 52
	),
	PosScore(
		"- - - - O O - -"
		"- - - O O O - -"
		"- X X X X O O O"
		"- - X X O O X O"
		"- X X X X X O O"
		"- - O O O X O O"
		"- - X - O X - O"
		"- - - - - X - -"_pos, -2 / 2 // 53
	),
	PosScore(
		"- - O O O - - -"
		"X X O O - - - -"
		"X X X X O O O O"
		"X X X X O X - -"
		"X X X O X X - -"
		"X X O O O - - -"
		"- - - O O O - -"
		"- - - O - - - -"_pos, -2 / 2 // 54
	),
	PosScore(
		"- - - - - - - -"
		"O - O - - - - -"
		"- O O O O X X X"
		"X X O X O O - -"
		"X X X O O O O -"
		"X X O O O O - -"
		"X - X X X O - -"
		"- - - X X - - -"_pos, +0 / 2 // 55
	),
	PosScore(
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O X O - -"
		"- X O O O O O -"
		"X X X X X O X -"
		"- - - X O O - -"
		"- - - - - - - -"_pos, +2 / 2 // 56
	),
	PosScore(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X O O O"
		"- - X X X O O O"
		"- - X X O X O O"
		"- O O O X X X O"
		"- - O X O O - O"
		"- O O O O O - -"_pos, -10 / 2 // 57
	),
	PosScore(
		"- - X O O O - -"
		"- - O O O - - -"
		"- O O O X O O -"
		"- O O O O X O -"
		"- O X O X X X -"
		"O O X X X X - -"
		"- - X - X X - -"
		"- - - - - - - -"_pos, +4 / 2 // 58
	),
	PosScore(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - O"
		"- - O O O O O -"
		"- - O O O O O X"
		"O O O O X X X X"
		"- - X X O O X X"
		"- - X X - O - X"_pos, +64 / 2 // 59
	),
	PosScore(
		"- - - O O O O -"
		"- - - O O O - -"
		"- - X O X O X X"
		"- - X O O X X X"
		"- - X O O X X X"
		"- - X O O O X X"
		"- - O X X X - X"
		"- - X X X X - -"_pos, +20 / 2 // 60
	),
	PosScore(
		"- O O O O - - -"
		"O - O O X O - -"
		"O O O O X O O -"
		"O X X O X X X X"
		"O X X X X X X -"
		"O O X X X X - -"
		"O - - - X - - -"
		"- - - - - - - -"_pos, -14 / 2 // 61
	),
	PosScore(
		"- - X X X X - -"
		"- - X X O O - -"
		"- - X O O O O O"
		"O O X O O X X X"
		"- O O O O X X -"
		"X O O O O O O X"
		"- - - - O - - -"
		"- - - - - - - -"_pos, +28 / 2 // 62
	),
	PosScore(
		"- - O - - - - -"
		"- - O - O - - -"
		"- X O O O O - -"
		"- X O O O O X -"
		"X X O X O X X X"
		"X X X X X O X -"
		"- - O X O O - -"
		"- - - O O O O -"_pos, -2 / 2 // 63
	),
	PosScore(
		"- - X - - O - -"
		"- - X - - O - X"
		"- X X O O O X X"
		"- - O O O O O X"
		"- - O O X X X X"
		"- O O O O O O -"
		"- - O O O - - -"
		"- O - X X X - -"_pos, +20 / 2 // 64
	),
	PosScore(
		"- - - - O O - -"
		"- - O O O O X -"
		"- - O X X X X -"
		"O - O X X X X -"
		"- O O X X O X -"
		"X X O X X X X -"
		"- - O O O O - -"
		"- - - - - O - -"_pos, +10 / 2 // 65
	),
	PosScore(
		"- O O O - - - -"
		"X - O X X - - -"
		"X X O X X O O -"
		"X O X X O O - -"
		"X X O O O O - -"
		"X X O O O O - -"
		"- - O O O - - -"
		"- - O - - - - -"_pos, +30 / 2 // 66
	),
	PosScore(
		"- X X X X X - -"
		"- - X O X X - -"
		"O O O X O X O -"
		"- O O O X O O O"
		"- O O O X X O -"
		"- - O O O X - O"
		"- - - O X - - -"
		"- - - - - - - -"_pos, +22 / 2 // 67
	),
	PosScore(
		"- - - O O O - -"
		"- - O O O O - -"
		"- - O X X O O X"
		"- O O X X O X -"
		"- O O X X X X -"
		"- X O O X X - -"
		"- - O O O - - -"
		"- - - - - O - -"_pos, +28 / 2 // 68
	),
	PosScore(
		"- - O O O O - -"
		"- - - O O O - -"
		"- O O O O O - -"
		"X X O X X O O -"
		"- O X O X O O -"
		"O X X X X X X -"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +0 / 2 // 69
	),
	PosScore(
		"- - - X - - - -"
		"X - X X X - - -"
		"X X X X - - - -"
		"X X X O O O - -"
		"X X X X O O - -"
		"X X O O X X X -"
		"X - O O X X - -"
		"- - O - - - - -"_pos, -24 / 2 // 70
	),
	PosScore(
		"- - - - - - - -"
		"- - - - - - - -"
		"- - O O O O O -"
		"- O O O O O X -"
		"- X O O O X X O"
		"- - X O X O O O"
		"- - X X O O - O"
		"- - - O O O O -"_pos, +20 / 2 // 71
	),
	PosScore(
		"- - - X - - - -"
		"- - X X O O - -"
		"- O O X O O O -"
		"O O O O X X O O"
		"- O O O O X X -"
		"- - O O O X X -"
		"- - - O O - - -"
		"- - - - O - - -"_pos, +24 / 2 // 72
	),
	PosScore(
		"- - O - - O - -"
		"- - O O O - - -"
		"X X O O O O - -"
		"- X X O O O - -"
		"- X O X O O X -"
		"X X X O O O O -"
		"X - - X O X - -"
		"- - - - - - - -"_pos, -4 / 2 // 73
	),
	PosScore(
		"- - - - O - - -"
		"- - X O O X - O"
		"- - X O X X O O"
		"- X X O O X O O"
		"- - X O O X - O"
		"- - O O X X - -"
		"- - O X X X - -"
		"- - - - - X - -"_pos, -30 / 2 // 74
	),
	PosScore(
		"- - - - O - - -"
		"- - - - O O - -"
		"- - X X O X - O"
		"- X X X O X O O"
		"- - O O O O O -"
		"- - O O O X O X"
		"- - O O O O X -"
		"- - - - - O - -"_pos, +14 / 2 // 75
	),
	PosScore(
		"- - - O - - - -"
		"- - O O - O - -"
		"- - - O O O X -"
		"O O O O O O X -"
		"- X X X X O X X"
		"- - O O O O O O"
		"- - O O O - - -"
		"- - - - O - - -"_pos, +32 / 2 // 76
	),
	PosScore(
		"- - O - O X - -"
		"X - O O O - - -"
		"X X O O O - - -"
		"X X O X O O O O"
		"- O O O O O - -"
		"O - X - O - - -"
		"- - O X - - - -"
		"- - - - - - - -"_pos, +34 / 2 // 77
	),
	PosScore(
		"- - - - O - - -"
		"- - O O O O - -"
		"- O O O X - X -"
		"O O X O X X X X"
		"- X O O X - - -"
		"X O O O - X - -"
		"- - O O - - - -"
		"- - - O - - - -"_pos, +8 / 2 // 78
	),
	PosScore(
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