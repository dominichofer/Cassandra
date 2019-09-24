#pragma once
#include "Moves.h"
#include "Position.h"

[[nodiscard]]
int CountLastFlip(Position, Field move);

[[nodiscard]]
Position Play(Position, Field move);

[[nodiscard]]
Position PlayPass(Position);

[[nodiscard]]
Moves PossibleMoves(Position);
