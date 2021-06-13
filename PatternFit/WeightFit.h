#pragma once
#include "Search/Puzzle.h"
#include <ranges>
#include <vector>

namespace WeightFit
{
    class Data
    {
        // Constraint: pos.size() == score.size()

        std::vector<Position> pos;
        std::vector<float> score;
    public:
        Data() noexcept = default;

        const std::vector<Position>& Pos() const noexcept { return pos; }
        const std::vector<float>& Score() const noexcept { return score; }

        void reserve(std::size_t new_capacity) { pos.reserve(new_capacity); score.reserve(new_capacity); }
        void push_back(const Position& pos, float score) { this->pos.push_back(pos); this->score.push_back(score); }
        std::size_t size() const { return pos.size(); }
        bool empty() const { return pos.empty(); }
    };

    struct DataPair
    {
        Data train;
        Data test;

        template <typename Iter>
        void Add(Iter first, Iter last, std::size_t train_size, std::size_t test_size) noexcept(false)
        {
            const std::size_t size = std::distance(first, last);

            if (train_size + test_size > size)
                throw std::runtime_error("train_size + test_size > size");

            train.reserve(train.size() + train_size);
            test.reserve(test.size() + test_size);

            Iter it1 = first;
            Iter it2 = it1 + train_size;
            Iter it3 = it2 + test_size;

            for (const Puzzle& p : std::ranges::subrange(it1, it2))
                if (p.HasTaskWithoutMove())
                    train.push_back(p.pos, p.MaxIntensity().result.score);

            for (const Puzzle& p : std::ranges::subrange(it2, it3))
                if (p.HasTaskWithoutMove())
                    test.push_back(p.pos, p.MaxIntensity().result.score);
        }
        template <typename Iter>
        void Add(Iter first, Iter last, double test_fraction) noexcept(false)
        {
            std::size_t size = std::distance(first, last);
            std::size_t test_size = test_fraction * size;
            std::size_t train_size = size - test_size;
            Add(first, last, train_size, test_size);
        }
        template <typename Container>
        void Add(const Container& c, std::size_t train_size, std::size_t test_size) noexcept(false)
        {
            Add(c.begin(), c.end(), train_size, test_size);
        }
        template <typename Container>
        void Add(const Container& c, double test_fraction) noexcept(false)
        {
            Add(c.begin(), c.end(), test_fraction);
        }
    };
}

Pattern::Weights FitWeights(const std::vector<BitBoard>& pattern, const WeightFit::Data&);

std::vector<float> EvalErrors(const Pattern::Weights&, const std::vector<BitBoard>& pattern, const WeightFit::Data&);

// Returns std::tuple{ StdDev(train_error), StdDev(test_error) }
std::tuple<double, double> EvalAccuracy(const Pattern::Weights&, const std::vector<BitBoard>& pattern, const WeightFit::DataPair&);