#pragma once
#include "cuda_runtime.h"
#include "Core/Position.h"

__host__ int64 perft_cuda(const Position& pos, int depth, int cuda_depth);