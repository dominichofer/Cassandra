#pragma once
#include "PosScore.h"
#include <vector>

static std::vector<PosScore> FForum =
{
	{ Position::Start(), +0 },
	{
		"- - X X X X X -"
		"- O O O X X - O"
		"- O O O X X O X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- X X X O O O -"
		"- O O O O O - -"_pos, +18 / 2 // 01
	},
	{
		"- X X X X X X -"
		"- - X O O O O -"
		"- X O X X O O X"
		"X O O O O O O O"
		"O X O O X X O O"
		"O O O X X O O X"
		"- O O O O O - -"
		"- - X X X X X -"_pos, +10 / 2 // 02
	},
	{
		"- - - - O X - -"
		"- - O O X X - -"
		"- O O O X X - X"
		"O O X X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X X O X O"
		"- - O O O O O X"_pos, +2 / 2 // 03
	},
	{
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X X O X O O O X"
		"- O X O O X X X"
		"- - O O O X X X"
		"- - O O X X - -"
		"- - X O X X O -"_pos, +0 / 2 // 04
	},
	{
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O X O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O O - -"
		"- X X X X X - -"_pos, +32 / 2 // 05
	},
	{
		"- - O X X X - -"
		"O O O X X X - -"
		"O O O X O X O -"
		"O O X O O O X -"
		"O O X X X X X X"
		"X O O X X O X -"
		"- O O O O X - -"
		"- X X X X X X -"_pos, +14 / 2 // 06
	},
	{
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O X X X X"
		"X O O X X X X X"
		"X O O O O X X X"
		"- X X X X X X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 07
	},
	{
		"- - - O - O - -"
		"O - O O O O - -"
		"O O O O X O O O"
		"O O O X X X X X"
		"O O X O O O X -"
		"O X O O O O X -"
		"O X X O O O - -"
		"O X X O O X - -"_pos, +8 / 2 // 08
	},
	{
		"- - O X O O - -"
		"X - X X O O O O"
		"- X X X O O O O"
		"- O X O O O X O"
		"O O X O X X X O"
		"X O O X O X O O"
		"- - X O X X - -"
		"- - X X X X - -"_pos, -8 / 2 // 09
	},
	{
		"- O O O O - - -"
		"- - X O O O - -"
		"O X O X O X O O"
		"X O X O O X O O"
		"X O O X O X X X"
		"O O O X O X X O"
		"- - X O O X - -"
		"- X X X X X - -"_pos, +10 / 2 // 10
	},
	{
		"- - - X - O X O"
		"- - - - O O X O"
		"- - - O O X X O"
		"X - O O X O X O"
		"O O O X X O X O"
		"- O X X O O O O"
		"O X X X O O - O"
		"X X X X X X X -"_pos, +30 / 2 // 11
	},
	{
		"- - X - - X - -"
		"O - X X X X O -"
		"O O X X X O X X"
		"O O X O X O X X"
		"O O X O O X X X"
		"O O O O X X X X"
		"- - X O O O - -"
		"- O O O O O - -"_pos, -8 / 2 // 12
	},
	{
		"- - X X X X X -"
		"- O O O X X - -"
		"- O O O X X X X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- - O X O O O -"
		"- O O O O O - -"_pos, +14 / 2 // 13
	},
	{
		"- - X X X X X -"
		"- - O O O X - -"
		"- X O O X X X X"
		"- O O O O O O O"
		"O O O X X X O O"
		"O O O X X O O X"
		"- - X X O O - -"
		"- - X X X X X -"_pos, +18 / 2 // 14
	},
	{
		"- - - - O - - -"
		"- - - O O X - -"
		"- O O O X X - X"
		"O O O X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X O O X O"
		"- - O O O O O X"_pos, +4 / 2 // 15
	},
	{
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X O O X X X O X"
		"- O O O X X X X"
		"- - O O X X X X"
		"- - - O O O - -"
		"- - X O X - O -"_pos, +24 / 2 // 16
	},
	{
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O O O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O - - -"
		"- X X X X - - -"_pos, +8 / 2 // 17
	},
	{
		"- X X X - - - -"
		"- - O O O X - -"
		"X O O O O O X X"
		"O X O X O O X X"
		"O X X O O O O O"
		"X X X O X O O X"
		"- - O X X O - -"
		"- O O O O O - -"_pos, -2 / 2 // 18
	},
	{
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O O X X X"
		"X O O O X X X X"
		"X - O O O X X X"
		"- - O O O O X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 / 2 // 19
	},
	{
		"X X X O X X X X"
		"O X X X X X X X"
		"O O X X X X X X"
		"O O O X X X X X"
		"O O O X X O O -"
		"O O O O O - - -"
		"O O O O O O O -"
		"O O O O O O O -"_pos, +6 / 2 // 20
	},
	{
		"X X X X X X X X"
		"O X X O O O - -"
		"O O X X O X X -"
		"O X O X X X - -"
		"O X X X X O - -"
		"O X X O X X - -"
		"O X X X X X - -"
		"O O O O - - - -"_pos, +0 / 2 // 21
	},
	{
		"- - X X X X - -"
		"O - X X X X X -"
		"O O X X O X O O"
		"O X O X O O O O"
		"O O O X O O O O"
		"- O O X O X O O"
		"- - X O O O - O"
		"- - - - O - - -"_pos, +2 / 2 // 22
	},
	{
		"- - O - - - - -"
		"- - O O X - - -"
		"O O O X X X O -"
		"O O O O X O X X"
		"X X X O O X O X"
		"X X X X X O O X"
		"X - X X X X O X"
		"- - X X X X - -"_pos, +4 / 2 // 23
	},
	{
		"- - X - - X - -"
		"- - - X X X O -"
		"- O - O X O X X"
		"- - O O O X X X"
		"O O O O X X X X"
		"O O O X O O X X"
		"O O O O O O - -"
		"O X O O - X - -"_pos, +0 / 2 // 24
	},
	{
		"- - - - O - - -"
		"- - - O O O X -"
		"- X X X O O O O"
		"O X X X X O O X"
		"- O O X X O O X"
		"X X O X O O O O"
		"X X X O O - - -"
		"O - O O O O - -"_pos, +0 / 2 // 25
	},
	{
		"- O O O O O - -"
		"- - O X X O - -"
		"- O O O O X X O"
		"- O O O X O X X"
		"- O O X O O X X"
		"- X O X X O X X"
		"- - O - X X X X"
		"- - O - - - - O"_pos, +0 / 2 // 26
	},
	{
		"- - X O - O - -"
		"- - O O O O - -"
		"O O X O X X O -"
		"O O O O X X O O"
		"O O O X X O X -"
		"O X O X X X X X"
		"- - X X X X - -"
		"- - X - O - X -"_pos, -2 / 2 // 27
	},
	{
		"- - O - - - - -"
		"- - O O O - - X"
		"- X O O O O X X"
		"X X X X O X O X"
		"- X X O X O O X"
		"X X O X O O X X"
		"- O O O O O - X"
		"- - - O O O - -"_pos, +0 / 2 // 28
	},
	{
		"- O X X X X - -"
		"- - O X X O - -"
		"X X O O X O O O"
		"X X X O O X O O"
		"X X O O X O O O"
		"X X X X O O - X"
		"X - X X O - - -"
		"- - - - - - - -"_pos, +10 / 2 // 29
	},
	{
		"- X X X - - - -"
		"X - X O O - - -"
		"X X O X O O - -"
		"X O X O X O - -"
		"X O O X O X X X"
		"X O O X X O X -"
		"- - O O O O O -"
		"- X X X X X - -"_pos, +0 / 2 // 30
	},
	{
		"- O O O O O - -"
		"- - O O O O - -"
		"O X X O O O - -"
		"- X X X O O - -"
		"X X X X X X O -"
		"X X X O O O - O"
		"X - O O O O - -"
		"- O O O O O - -"_pos, -2 / 2 // 31
	},
	{
		"- - X X - - - -"
		"O - X X O X - -"
		"O O X O O - - -"
		"O X O X O O O -"
		"O O X X O O O X"
		"O O X X X O O X"
		"- - X X X X O X"
		"- - X - - X - X"_pos, -4 / 2 // 32
	},
	{
		"- X X X X X X X"
		"- - X O O O - -"
		"- - O X O O X X"
		"- O O X X O X X"
		"- O O O O O X X"
		"- X - X O O X X"
		"- - - O - X - X"
		"- - O O O O - -"_pos, -8 / 2 // 33
	},
	{
		"- - - - - - - -"
		"- - - - - O - O"
		"- O O O O O O O"
		"O O O O O X O O"
		"O X X O O O X O"
		"- X X X O X O O"
		"- - X X X O X O"
		"- - O X X X X O"_pos, -2 / 2 // 34
	},
	{
		"- - O O O - - -"
		"- - O O O O - X"
		"X X O O X X X X"
		"X X X X X X O X"
		"- X X O O O O X"
		"- X X X O O O X"
		"- - - O X O O -"
		"- - O - - - - -"_pos, +0 / 2 // 35
	},
	{
		"- - - O - X - -"
		"- - O O O X - O"
		"O O O O O O O O"
		"O X X O O X X O"
		"O X O X X X O O"
		"O O X X X X - O"
		"O - - X X X X -"
		"- - - - - - - -"_pos, +0 / 2 // 36
	},
	{
		"- - O O O O - -"
		"O - O O O O - -"
		"O X X X O O O -"
		"O X X O X O - -"
		"O O X X O X X -"
		"O O X X X X - -"
		"O - X X X - - -"
		"- - X X - O - -"_pos, -20 / 2 // 37
	},
	{
		"- - O O O O - -"
		"- - O O O O - -"
		"- X O X X O O X"
		"O O X O O O O X"
		"- O O O O O X X"
		"X O O X X X X X"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +4 / 2 // 38
	},
	{
		"X - X X X X - -"
		"O X O O X O - -"
		"O X X X O O O -"
		"O X X X O O - -"
		"O X X O X O - -"
		"O X O O O - - -"
		"O - O O - - - -"
		"- - - - - - - -"_pos, +64 / 2 // 39
	},
	{
		"O - - O O O O X"
		"- O O O O O O X"
		"O O X X O O O X"
		"O O X O O O X X"
		"O O O O O O X X"
		"- - - O O O O X"
		"- - - - O - - X"
		"- - - - - - - -"_pos, +38 / 2 // 40
	},
	{
		"- O O O O O - -"
		"- - O O O O X -"
		"- O O O O O O -"
		"X X X X X O O -"
		"- X X O O X - -"
		"O O X O X X - -"
		"- - O X X O - -"
		"- O O O - - O -"_pos, +0 / 2 // 41
	},
	{
		"- - O O O - - -"
		"- - - - X X - O"
		"O O O O O X O O"
		"- O O O O X O O"
		"X - O O O X X O"
		"- - - O O X O O"
		"- - - O O O X O"
		"- - O O O O - -"_pos, +6 / 2 // 42
	},
	{
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O O O O -"
		"- X X O O O X -"
		"X X X X O X X -"
		"- - - O X O - -"
		"- - O O O O O -"_pos, -12 / 2 // 43
	},
	{
		"- - X - O - X -"
		"- - X - O X - X"
		"- X X O O O X X"
		"X X X X O O O X"
		"X X X X O O - -"
		"O O X X O X - -"
		"- - O O O O - -"
		"- - - O O O - -"_pos, -14 / 2 // 44
	},
	{
		"- - - X X X X -"
		"X - X X X O - -"
		"X X O X O O - -"
		"X X X O X O - -"
		"X X O X X O - -"
		"- O X X X O O -"
		"O - O O O O - -"
		"- - - - O O - -"_pos, +6 / 2 // 45
	},
	{
		"- - - X X X - -"
		"- - O O O X - -"
		"- - O O O X X -"
		"- O O O O X X X"
		"- - O O O O X X"
		"- - O X O X X X"
		"- - X X O O - -"
		"- X X X X - O -"_pos, -8 / 2 // 46
	},
	{
		"- X X X X X - -"
		"- - X X X X - -"
		"- X X X X O - -"
		"O O O O O O - -"
		"- X O X X O - -"
		"X X X O X O - -"
		"- - X X O O - -"
		"- - O O O O - -"_pos, +4 / 2 // 47
	},
	{
		"- - - - - O - -"
		"O - O O O - - -"
		"O O O O X X - -"
		"O X O X X O O -"
		"O X X O O O - -"
		"O X X O O - - -"
		"- - X X X O - -"
		"- O O O O O O -"_pos, +28 / 2 // 48
	},
	{
		"- - O X - O - -"
		"- - X X O O - -"
		"O O O O O X X -"
		"O O O O O X - -"
		"O O O X O X X -"
		"O O O O X X - -"
		"- - - O O X - -"
		"- - X - O - - -"_pos, +16 / 2 // 49
	},
	{
		"- - - - X - - -"
		"- - X X X - - -"
		"- O O O X O O O"
		"- O O O X O O O"
		"- O X O X O X O"
		"- O O X X O O O"
		"- - O O X O - -"
		"- - O - - O - -"_pos, +10 / 2 // 50
	},
	{
		"- - - - X - O -"
		"- - - - - O - -"
		"- - - O O O X -"
		"X O O O O O X X"
		"- O O X X O X X"
		"O O X O O O X X"
		"- - X X X X - X"
		"- - - - X X - -"_pos, +6 / 2 // 51
	},
	{
		"- - - O - - - -"
		"- - - X O - - O"
		"- - O X X O O O"
		"O O O X O O O O"
		"O O O X X O O O"
		"O O O X X X O O"
		"- - O X - - - O"
		"- - - - - - - -"_pos, +0 / 2 // 52
	},
	{
		"- - - - O O - -"
		"- - - O O O - -"
		"- X X X X O O O"
		"- - X X O O X O"
		"- X X X X X O O"
		"- - O O O X O O"
		"- - X - O X - O"
		"- - - - - X - -"_pos, -2 / 2 // 53
	},
	{
		"- - O O O - - -"
		"X X O O - - - -"
		"X X X X O O O O"
		"X X X X O X - -"
		"X X X O X X - -"
		"X X O O O - - -"
		"- - - O O O - -"
		"- - - O - - - -"_pos, -2 / 2 // 54
	},
	{
		"- - - - - - - -"
		"O - O - - - - -"
		"- O O O O X X X"
		"X X O X O O - -"
		"X X X O O O O -"
		"X X O O O O - -"
		"X - X X X O - -"
		"- - - X X - - -"_pos, +0 / 2 // 55
	},
	{
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O X O - -"
		"- X O O O O O -"
		"X X X X X O X -"
		"- - - X O O - -"
		"- - - - - - - -"_pos, +2 / 2 // 56
	},
	{
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X O O O"
		"- - X X X O O O"
		"- - X X O X O O"
		"- O O O X X X O"
		"- - O X O O - O"
		"- O O O O O - -"_pos, -10 / 2 // 57
	},
	{
		"- - X O O O - -"
		"- - O O O - - -"
		"- O O O X O O -"
		"- O O O O X O -"
		"- O X O X X X -"
		"O O X X X X - -"
		"- - X - X X - -"
		"- - - - - - - -"_pos, +4 / 2 // 58
	},
	{
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - O"
		"- - O O O O O -"
		"- - O O O O O X"
		"O O O O X X X X"
		"- - X X O O X X"
		"- - X X - O - X"_pos, +64 / 2 // 59
	},
	{
		"- - - O O O O -"
		"- - - O O O - -"
		"- - X O X O X X"
		"- - X O O X X X"
		"- - X O O X X X"
		"- - X O O O X X"
		"- - O X X X - X"
		"- - X X X X - -"_pos, +20 / 2 // 60
	},
	{
		"- O O O O - - -"
		"O - O O X O - -"
		"O O O O X O O -"
		"O X X O X X X X"
		"O X X X X X X -"
		"O O X X X X - -"
		"O - - - X - - -"
		"- - - - - - - -"_pos, -14 / 2 // 61
	},
	{
		"- - X X X X - -"
		"- - X X O O - -"
		"- - X O O O O O"
		"O O X O O X X X"
		"- O O O O X X -"
		"X O O O O O O X"
		"- - - - O - - -"
		"- - - - - - - -"_pos, +28 / 2 // 62
	},
	{
		"- - O - - - - -"
		"- - O - O - - -"
		"- X O O O O - -"
		"- X O O O O X -"
		"X X O X O X X X"
		"X X X X X O X -"
		"- - O X O O - -"
		"- - - O O O O -"_pos, -2 / 2 // 63
	},
	{
		"- - X - - O - -"
		"- - X - - O - X"
		"- X X O O O X X"
		"- - O O O O O X"
		"- - O O X X X X"
		"- O O O O O O -"
		"- - O O O - - -"
		"- O - X X X - -"_pos, +20 / 2 // 64
	},
	{
		"- - - - O O - -"
		"- - O O O O X -"
		"- - O X X X X -"
		"O - O X X X X -"
		"- O O X X O X -"
		"X X O X X X X -"
		"- - O O O O - -"
		"- - - - - O - -"_pos, +10 / 2 // 65
	},
	{
		"- O O O - - - -"
		"X - O X X - - -"
		"X X O X X O O -"
		"X O X X O O - -"
		"X X O O O O - -"
		"X X O O O O - -"
		"- - O O O - - -"
		"- - O - - - - -"_pos, +30 / 2 // 66
	},
	{
		"- X X X X X - -"
		"- - X O X X - -"
		"O O O X O X O -"
		"- O O O X O O O"
		"- O O O X X O -"
		"- - O O O X - O"
		"- - - O X - - -"
		"- - - - - - - -"_pos, +22 / 2 // 67
	},
	{
		"- - - O O O - -"
		"- - O O O O - -"
		"- - O X X O O X"
		"- O O X X O X -"
		"- O O X X X X -"
		"- X O O X X - -"
		"- - O O O - - -"
		"- - - - - O - -"_pos, +28 / 2 // 68
	},
	{
		"- - O O O O - -"
		"- - - O O O - -"
		"- O O O O O - -"
		"X X O X X O O -"
		"- O X O X O O -"
		"O X X X X X X -"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +0 / 2 // 69
	},
	{
		"- - - X - - - -"
		"X - X X X - - -"
		"X X X X - - - -"
		"X X X O O O - -"
		"X X X X O O - -"
		"X X O O X X X -"
		"X - O O X X - -"
		"- - O - - - - -"_pos, -24 / 2 // 70
	},
	{
		"- - - - - - - -"
		"- - - - - - - -"
		"- - O O O O O -"
		"- O O O O O X -"
		"- X O O O X X O"
		"- - X O X O O O"
		"- - X X O O - O"
		"- - - O O O O -"_pos, +20 / 2 // 71
	},
	{
		"- - - X - - - -"
		"- - X X O O - -"
		"- O O X O O O -"
		"O O O O X X O O"
		"- O O O O X X -"
		"- - O O O X X -"
		"- - - O O - - -"
		"- - - - O - - -"_pos, +24 / 2 // 72
	},
	{
		"- - O - - O - -"
		"- - O O O - - -"
		"X X O O O O - -"
		"- X X O O O - -"
		"- X O X O O X -"
		"X X X O O O O -"
		"X - - X O X - -"
		"- - - - - - - -"_pos, -4 / 2 // 73
	},
	{
		"- - - - O - - -"
		"- - X O O X - O"
		"- - X O X X O O"
		"- X X O O X O O"
		"- - X O O X - O"
		"- - O O X X - -"
		"- - O X X X - -"
		"- - - - - X - -"_pos, -30 / 2 // 74
	},
	{
		"- - - - O - - -"
		"- - - - O O - -"
		"- - X X O X - O"
		"- X X X O X O O"
		"- - O O O O O -"
		"- - O O O X O X"
		"- - O O O O X -"
		"- - - - - O - -"_pos, +14 / 2 // 75
	},
	{
		"- - - O - - - -"
		"- - O O - O - -"
		"- - - O O O X -"
		"O O O O O O X -"
		"- X X X X O X X"
		"- - O O O O O O"
		"- - O O O - - -"
		"- - - - O - - -"_pos, +32 / 2 // 76
	},
	{
		"- - O - O X - -"
		"X - O O O - - -"
		"X X O O O - - -"
		"X X O X O O O O"
		"- O O O O O - -"
		"O - X - O - - -"
		"- - O X - - - -"
		"- - - - - - - -"_pos, +34 / 2 // 77
	},
	{
		"- - - - O - - -"
		"- - O O O O - -"
		"- O O O X - X -"
		"O O X O X X X X"
		"- X O O X - - -"
		"X O O O - X - -"
		"- - O O - - - -"
		"- - - O - - - -"_pos, +8 / 2 // 78
	},
	{
		"- - - - - - - -"
		"- - - - - - X -"
		"- - - - O - X X"
		"- - - O O O X -"
		"O O O O X O X X"
		"- - O O O O O O"
		"- - O - O O - O"
		"- - - - O O - -"_pos, +64 / 2 // 79
	},
};
