#pragma once
#include <cstdint>

uint64_t FlippedCodiagonal(uint64_t) noexcept;
uint64_t FlippedDiagonal(uint64_t) noexcept;
uint64_t FlippedHorizontal(uint64_t) noexcept;
uint64_t FlippedVertical(uint64_t) noexcept;

bool IsCodiagonallySymmetric(uint64_t) noexcept;
bool IsDiagonallySymmetric(uint64_t) noexcept;
bool IsHorizontallySymmetric(uint64_t) noexcept;
bool IsVerticallySymmetric(uint64_t) noexcept;

uint64_t ParityQuadrants(uint64_t) noexcept;

uint64_t EightNeighboursAndSelf(uint64_t) noexcept;