#pragma once
#include "PVS_Node.h"
#include <functional>
#include <mutex>
#include <optional>

template <typename Position, typename Result>
class ParallelTree
{
	YBWC_Node<Position, Result> root;
	std::function<void(YBWC_Node<Position, Result>&)> expand;
	std::function<Result(YBWC_Node<Position, Result>&)> summarize;
	std::mutex mtx;
	std::condition_variable out_of_work;
public:
	ParallelTree(
		const Position& pos,
		OpenInterval window,
		std::function<void(YBWC_Node<Position, Result>&)> expand,
		std::function<Result(YBWC_Node<Position, Result>&)> summarize
	) noexcept
		: root(YBWC_Node<Position, Result>{ pos, window })
		, expand(expand)
		, summarize(summarize)
	{}

	bool is_solved() const { return root.is_solved(); }
	Result result() const { return root.result; }

	template <typename Evaluator>
	void work_on(Evaluator&& eval, std::function<void()> stop_callback = [] {})
	{
		std::atomic<bool> stopped;
		auto extended_stop_callback = [&]() { stopped.store(true, std::memory_order_release); stop_callback(); };
		std::unique_lock lock{ mtx };
		while (not is_solved())
		{
			YBWC_Node<Position, Result>* node = next_unsolved_leaf_node(root, expand);
			if (node)
			{
				stopped.store(false, std::memory_order_release);
				node->block(extended_stop_callback);
				Position pos = node->pos;
				OpenInterval window = node->window;
				lock.unlock();
				Result result;
				try
				{
					result = eval(pos, window);
				}
				catch (...)
				{}
				lock.lock();
				if (not stopped.load(std::memory_order_acquire))
				{
					node->solve(result, summarize);
					out_of_work.notify_all();
				}
			}
			else
				out_of_work.wait(lock);
		}
	}
};
