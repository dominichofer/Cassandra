#include "PatternFit.h"
#include "Pattern/Indexer.h"
#include "Math/Math.h"



//void EvalForAccuracyFit(range<NoMovePuzzle> auto& data, const Algorithm& algorithm, int max_depth)
//{
//    Process(
//        std::execution::par,
//        data,
//        [&](NoMovePuzzle& p) {
//            p.erase_inexacts();
//            for (int d = 0; d <= std::min(max_depth, p.EmptyCount()); d++)
//                p.insert(Request(d));
//            p.Solve(algorithm);
//        }
//    );
//}
//
//void Improve(
//    AAGLEM& model,
//    const random_access_range<NoMovePuzzle> auto& weight_fit_data,
//    range<NoMovePuzzle> auto& accuracy_model_data,
//    const Algorithm& algorithm,
//    int max_depth)
//{
//    for (int block = 0; block < model.Blocks(); block++)
//    {
//        auto [lower, upper] = model.Boundaries(block);
//        auto in_block = std::views::filter([&](const NoMovePuzzle& p) { return lower <= p.EmptyCount() and p.EmptyCount() < upper; });
//        auto weight_fit_block_data = weight_fit_data | in_block;
//        auto accuracy_model_block_data = accuracy_model_data | in_block;
//
//        model.SetWeights(block, FitWeights(weight_fit_block_data, model.Pattern()));
//        EvalForAccuracyFit(accuracy_model_block_data, algorithm, max_depth);
//        ImproveAccuracyModel(model, accuracy_model_block_data);
//    }
//}