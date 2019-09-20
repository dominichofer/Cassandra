#pragma once
#include "Moves.h"
#include "Position.h"
#include <cstdint>
#include <cstddef>

int CountLastFlip(Position pos, Field move);

Position Play(Position pos, Field move);

Position PlayPass(Position pos);

Moves PossibleMoves(Position pos);
