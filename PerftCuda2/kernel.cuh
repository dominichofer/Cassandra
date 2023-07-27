#pragma once
#include "cuda_runtime.h"
#include "Board/Board.h"
#include <cstdint>

__host__ int64_t perft_cuda(const Position& pos, int depth, int cuda_depth);
