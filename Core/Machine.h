#pragma once
#include "Core/Moves.h"
#include "Core/Position.h"

[[nodiscard]]
int CountLastFlip(const Position&, Field move);

[[nodiscard]]
BitBoard Flips(const Position&, Field move);

[[nodiscard]]
Position Play(const Position&, Field move, BitBoard flips);

[[nodiscard]]
Position Play(const Position&, Field move);

[[nodiscard]]
Position PlayPass(const Position&);

[[nodiscard]]
Moves PossibleMoves(const Position&);

[[nodiscard]]
Field GetFirstDisc(BitBoard);


// Stable stones of the opponent
[[nodiscard]]
BitBoard StableStones(const Position&);


// Forward declarations

[[nodiscard]]
std::size_t CountTrailingZeros(uint64_t) noexcept;

[[nodiscard]]
std::size_t BitScanLSB(uint64_t) noexcept;

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
