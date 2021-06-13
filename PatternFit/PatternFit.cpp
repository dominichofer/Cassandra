#include "PatternFit.h"
#include "Search/Puzzle.h"
#include "IO/IO.h"
#include <ranges>
#include <iostream>
#include <vector>

void FitWeights()
{
    const double test_fraction = 0.01;
    const auto pattern = LoadPattern();

    for (std::size_t block = 0; block < 5; block++)
    {
        std::size_t expected_train_size = 0;
        WeightFit::DataPair data;
        for (std::size_t e = 1 + PatternEval::block_size * block; e <= PatternEval::block_size * (block + 1); e++)
        {
            auto puzzles = Load<std::vector<Puzzle>>(R"(G:\Reversi\rnd_100k\e)" + std::to_string(e) + ".puz");
            data.Add(puzzles, test_fraction);
        }

        Pattern::Weights weights = FitWeights(pattern, data.train);
        //if (something fishy happened when fitting) TODO!
        //    weights = LoadWeights(block - 1);
        SaveWeights(weights, block);

        auto [train_sd, test_sd] = EvalAccuracy(weights, pattern, data);
        std::cout << "Block " << block << ": train error " << train_sd << ", test error " << test_sd << "\n";
    }
}
