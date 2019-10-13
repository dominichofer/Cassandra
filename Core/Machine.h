#pragma once
#include "Core/Moves.h"
#include "Core/Position.h"

[[nodiscard]]
int CountLastFlip(Position, Field move);

[[nodiscard]]
BitBoard Flips(Board, Field move);

[[nodiscard]]
Board Play(Board, Field move, BitBoard flips);

[[nodiscard]]
Position Play(Position, Field move);

[[nodiscard]]
Position PlayPass(Position);

[[nodiscard]]
Moves PossibleMoves(Position);


// Forward declarations

[[nodiscard]]
unsigned int BitScanLSB(uint64_t) noexcept;

[[nodiscard]]
uint64_t GetLSB(uint64_t) noexcept;

void RemoveLSB(uint64_t&) noexcept;

[[nodiscard]]
std::size_t PopCount(uint64_t) noexcept;

[[nodiscard]]
uint64_t PDep(uint64_t src, uint64_t mask) noexcept;

[[nodiscard]]
uint64_t PExt(uint64_t src, uint64_t mask) noexcept;

[[nodiscard]]
uint64_t BSwap(uint64_t) noexcept;
