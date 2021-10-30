#pragma once
#include "IO/IO.h"
#include "Math/MatrixCSR.h"
#include "Math/Solver.h"
#include "Math/Statistics.h"
#include "Pattern/DenseIndexer.h"
#include <vector>

class PositionScore
{
    // Constraint: pos.size() == score.size()
    std::vector<Position> pos;
    std::vector<float> score;
public:
    PositionScore() noexcept = default;

    const std::vector<Position>& Pos() const noexcept { return pos; }
    const std::vector<float>& Score() const noexcept { return score; }

    void push_back(const Position& pos, float score) { this->pos.push_back(pos); this->score.push_back(score); }
    void reserve(std::size_t new_capacity) { pos.reserve(new_capacity); score.reserve(new_capacity); }
    std::size_t size() const { return pos.size(); }
    bool empty() const { return pos.empty(); }
    void clear() { pos.clear(); score.clear(); }
};

class TrainAndTest_PositionScore
{
public:
    PositionScore train;
    PositionScore test;

    void Add(PuzzleRange auto&& puzzle, double test_fraction) noexcept(false)
    {
        std::size_t size = std::distance(puzzle.begin(), puzzle.end());
        std::size_t test_size = size * test_fraction;
        std::size_t train_size = size - test_size;

        train.reserve(train.size() + train_size);
        test.reserve(test.size() + test_size);
        for (const Puzzle& p : puzzle | std::views::take(train_size))
            train.push_back(p.pos, p.MaxSolvedIntensityScore().value_or(0));

        for (const Puzzle& p : puzzle | std::views::drop(train_size) | std::views::take(test_size))
            test.push_back(p.pos, p.MaxSolvedIntensityScore().value_or(0));
    }

    void clear() { train.clear(); test.clear(); }
};

class WeightFitter
{
    TrainAndTest_PositionScore data;
    std::vector<BitBoard> pattern;
    std::vector<float> weights, train_error, test_error;
public:
    WeightFitter() noexcept = default;
    template <PuzzleRange Range>
    WeightFitter(std::vector<BitBoard> pattern, Range&& puzzle, double test_fraction)
        : pattern(std::move(pattern))
    {
        AddData(puzzle, test_fraction);
    }

    void Pattern(std::vector<BitBoard> pattern) { this->pattern = std::move(pattern); }
    const std::vector<BitBoard>& Pattern() const { return pattern; }

    void AddData(PuzzleRange auto&& puzzle, double test_fraction)
    {
        // Apply test_fraction to each EmptyCount.
        for (int e = 0; e <= 64; e++)
            data.Add(puzzle | std::views::filter([e](const Puzzle& p) { return p.pos.EmptyCount() == e; }), test_fraction);
    }
    void ClearData() { data.clear(); }
    void SetData(PuzzleRange auto&& puzzle, double test_fraction) { ClearData(); AddData(puzzle, test_fraction); }

    std::size_t TrainSize() const { return data.train.size(); }
    std::size_t TestSize() const { return data.test.size(); }
    std::size_t size() const { return TrainSize() + TestSize(); }

    void Fit();

    const std::vector<float>& Weights() const { return weights; }
    const std::vector<float>& TrainError() const { return train_error; }
    const std::vector<float>& TestError() const { return test_error; }
};

void FitWeights(const DataBase<Puzzle>& data, std::vector<BitBoard> pattern, int block_size, int block, bool print_results = false);
void FitWeights(const DataBase<Puzzle>& data, int block, bool print_results = false);
void FitWeights(const DataBase<Puzzle>& data, std::vector<BitBoard> pattern, int block_size, bool print_results = false);
void FitWeights(const DataBase<Puzzle>& data, bool print_results = false);

void EvaluateAccuracyFit(DataBase<Puzzle>& data, HashTablePVS&, AAGLEM&);
void FitAccuracyModel(const DataBase<Puzzle>& data, const AAGLEM&);

void FitPattern(const DataBase<Puzzle>& eval_fit, DataBase<Puzzle>& accuracy_fit, HashTablePVS&, AAGLEM&, std::vector<BitBoard> pattern, int block_size);
void FitPattern(const DataBase<Puzzle>& eval_fit, DataBase<Puzzle>& accuracy_fit, HashTablePVS&, AAGLEM&);