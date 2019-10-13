#pragma once
#include "Puzzle.h"
#include <vector>

struct PosScore
{
	Position pos;
	Score score;
};

static std::vector<PosScore> FForum =
{
	{ Position::Start() },
	{ Position{ BitBoard{ 0x3E0C0D2B772B7000ui64 }, BitBoard{ 0x0071725488140E7Cui64 } }, +18 },
	{ Position{ BitBoard{ 0x7E2059000C19303Eui64 }, BitBoard{ 0x001E267FF3E60C00ui64 } }, +10 },
	{ Position{ BitBoard{ 0x040C0D306C707A01ui64 }, BitBoard{ 0x083070CF938F853Eui64 } }, +2 },
	{ Position{ BitBoard{ 0x7EB8B9D127070C2Cui64 }, BitBoard{ 0x0006462E58383012ui64 } }, +0 },
	{ Position{ BitBoard{ 0x0019D6D6C8F0A07Cui64 }, BitBoard{ 0x7C242829370D1C00ui64 } }, +32 },
	{ Position{ BitBoard{ 0x1C1C14223F9A047Eui64 }, BitBoard{ 0x20E0EADCC0647800ui64 } }, +14 },
	{ Position{ BitBoard{ 0x18BC8F9F877F3018ui64 }, BitBoard{ 0x2440706078000E06ui64 } }, +8 },
	{ Position{ BitBoard{ 0x0000081F22426064ui64 }, BitBoard{ 0x14BCF7E0DCBC9C98ui64 } }, +8 },
	{ Position{ BitBoard{ 0x10B070222E942C3Cui64 }, BitBoard{ 0x2C0F0F5DD16B1000ui64 } }, -8 },
	{ Position{ BitBoard{ 0x002054A49716247Cui64 }, BitBoard{ 0x781CAB5B68E91800ui64 } }, +10 },
	// TODO: Add the others!
};
