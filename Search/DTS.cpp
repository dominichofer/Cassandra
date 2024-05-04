//#include "DynamicTreeSplit/PVS_Node.h"
//#include "DTS.h"
//#include "Status.h"
//#include <algorithm>
//#include <string>
//#include <ranges>
//
//struct PosDepthCL
//{
//    Position pos;
//    int depth;
//    ConfidenceLevel level;
//};
//
//inline bool operator>(const Result& result, const OpenInterval& i) noexcept { return result.score >= i.upper; }
//
//class ParallelTree
//{
//    std::mutex mtx;
//    YBWC_Node<PosDepthCL, Result> root;
//    std::function<void(YBWC_Node<PosDepthCL, Result>&)> expand;
//    std::function<Result(YBWC_Node<PosDepthCL, Result>&)> summarize;
//    std::condition_variable out_of_work;
//public:
//    ParallelTree(
//        const Position& pos,
//        int depth,
//        ConfidenceLevel level,
//        OpenInterval window,
//        std::function<void(YBWC_Node<PosDepthCL, Result>&)> expand,
//        std::function<Result(YBWC_Node<PosDepthCL, Result>&)> summarize
//    ) noexcept
//        : root(YBWC_Node<PosDepthCL, Result>{ PosDepthCL{ pos, depth, level }, window })
//        , expand(expand)
//        , summarize(summarize)
//    {}
//
//    bool is_solved() const { return root.is_solved(); }
//    Result result() const { return root.result; }
//
//    template <typename Evaluator>
//    void work_on(Evaluator&& eval, std::function<void()> stop_callback = [] {})
//    {
//        std::atomic<bool> stopped;
//        auto extended_stop_callback = [&]() { stopped.store(true, std::memory_order_release); stop_callback(); };
//        std::unique_lock lock{ mtx };
//        while (not is_solved())
//        {
//            YBWC_Node<PosDepthCL, Result>* node = next_unsolved_leaf_node(root, expand);
//            if (node)
//            {
//                stopped.store(false, std::memory_order_release);
//                node->block(extended_stop_callback);
//                Position pos = node->pos.pos;
//                int depth = node->pos.depth;
//                ConfidenceLevel level = node->pos.level;
//                OpenInterval window = node->window;
//                lock.unlock();
//                Result result;
//                try
//                {
//                    //std::cout << to_string(pos) << " e" << pos.EmptyCount() << " d" << depth << " " << to_string(window) << std::endl;
//                    result = eval(pos, depth, level, window);
//                }
//                catch (...)
//                {
//                }
//                lock.lock();
//                if (not stopped.load(std::memory_order_acquire))
//                {
//                    node->solve(result, summarize);
//                    out_of_work.notify_all();
//                }
//            }
//            else
//                out_of_work.wait(lock);
//        }
//    }
//};
//
//
//DTS::DTS(HT& tt, Algorithm& alg, MoveSorter& move_sorter, int parallel_plies) noexcept
//	: tt(tt)
//    , alg(alg)
//	, move_sorter(move_sorter)
//	, parallel_plies(parallel_plies)
//{}
//
//Result DTS::Eval(const Position& pos, OpenInterval window, Intensity intensity)
//{
//	if (depth < 18)
//		return alg.Eval(pos, window, depth, level);
//
//	int parallel_depth = depth - parallel_plies;
//	auto expand = [this, parallel_depth](YBWC_Node<PosDepthCL, Result>& node)
//	{
//		if (node.pos.depth < 16)
//			return;
//		if (PossibleMoves(node.pos.pos)) {
//			for (Field move : move_sorter.Sorted(node.pos.pos, node.pos.depth, node.pos.level))
//                node.append({ Play(node.pos.pos, move), node.pos.depth - 1, node.pos.level });
//		}
//		else {
//			auto passed = PlayPass(node.pos.pos);
//			if (PossibleMoves(passed))
//                node.append({ passed, node.pos.depth, node.pos.level });
//		}
//	};
//    auto summarize = [this](YBWC_Node<PosDepthCL, Result>& node)
//    {
//        Status status{ node.window.lower };
//        for (const YBWC_Node<PosDepthCL, Result>& child : node.children)
//            if (child.is_solved())
//                status.Update(child.result, static_cast<Field>(std::countr_zero(child.pos.pos.Discs() ^ node.pos.pos.Discs())));
//        Result r = status.GetResult();
//        tt.Insert(node.pos.pos, r);
//        return r;
//    };
//	auto eval = [this](const Position& pos, Intensity intensity, OpenInterval window) {
//		return alg.Eval(pos, window, depth, level);
//	};
//
//	ParallelTree tree(pos, depth, level, window, expand, summarize);
//
//	#pragma omp parallel
//	tree.work_on(eval);
//
//	return tree.result();
//}