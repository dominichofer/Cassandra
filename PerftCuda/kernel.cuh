#pragma once
#include "Board/Board.h"
#include <cstdint>

extern "C" __declspec(dllexport)
int64_t perft_cuda(const Position&, int depth);
