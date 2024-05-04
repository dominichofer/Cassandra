#include "kernel.cuh"
#include "device_launch_parameters.h"
#include "Board/Board.h"
#include "DeviceVector.cuh"
#include "HostVector.cuh"
#include <cstdint>
#include <omp.h>
#include <numeric>

#include "Board/Position.cpp"
#include "Board/PossibleMoves.cpp"
#include "Board/Field.cpp"
#include "Board/Flips.cpp"

// perft for 0 plies left
 __device__ int64_t perft_0()
{
    return 1;
}

// perft for 1 ply left
__device__ int64_t perft_1(Position pos)
{
    auto moves = PossibleMoves(pos);
    if (moves)
        return moves.size();
    return PossibleMoves(PlayPass(pos)) ? 1 : 0;
}

// perft for 2 plies left
__device__ int64_t perft_2(Position pos)
{
    auto moves = PossibleMoves(pos);
    if (!moves)
        return PossibleMoves(PlayPass(pos)).size();

    int64_t sum = 0;
    for (Field move : moves)
        sum += perft_1(Play(pos, move));
    return sum;
}

// perft for 3 plies left
__device__ int64_t perft_3(Position pos)
{
    auto moves = PossibleMoves(pos);
    if (!moves)
    {
        Position passed = PlayPass(pos);
        if (PossibleMoves(passed))
            return perft_2(passed);
        return 0;
    }

    int64_t sum = 0;
    for (Field move : moves)
        sum += perft_2(Play(pos, move));
    return sum;
}

__global__ void perft_3(const DeviceVectorView<Position> pos, DeviceVectorView<int64_t> result)
{
    unsigned int gridSize = blockDim.x * gridDim.x;
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < pos.size(); i += gridSize)
        result[i] = perft_3(pos[i]);
}

int64_t perft_cuda(const Position& pos, int depth)
{
    const int cuda_depth = 3;
    assert(depth > cuda_depth);

    static thread_local int tid = []() { int n; cudaGetDeviceCount(&n); int tid = omp_get_thread_num() % n; cudaSetDevice(tid); return tid; }();
    static thread_local PinnedVector<Position> positions;
    static thread_local PinnedVector<int64_t> result;
    static thread_local DeviceVector<Position> cuda_pos;
    static thread_local DeviceVector<int64_t> cuda_result;

    auto gen = Children(pos, depth - cuda_depth, true);
    positions.store(gen.begin(), gen.end());

    cuda_pos.store(positions, asyn);
    cuda_result.resize(positions.size(), asyn);
    perft_3<<<256, 128>>>(cuda_pos, cuda_result);
    result.store(cuda_result, asyn);
    cudaDeviceSynchronize();

    return std::accumulate(result.begin(), result.end(), 0i64);
}