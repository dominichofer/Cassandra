#include "pch.h"
#include <string>

TEST(ParallelTree, single_threaded)
{
	auto expand = [](auto& node) {
		if (node.pos.length() < 3)
			for (int i = 0; i < 10; i++)
				node.append(node.pos + std::to_string(i));
		};
	auto summarize = [](YBWC_Node<std::string, int>& node) { return 0; };
	auto eval = [](std::string pos, OpenInterval) { return static_cast<int>(std::stoul(pos, nullptr, 16)); };
	ParallelTree<std::string, int> tree{ "0", { -1000000, 1000000 }, expand, summarize };

	tree.work_on(eval);

	EXPECT_TRUE(tree.is_solved());
}

TEST(ParallelTree, multi_threaded)
{
	std::atomic_flag stop;
	auto expand = [](auto& node) {
		if (node.pos.length() < 3)
			for (int i = 0; i < 10; i++)
				node.append(node.pos + std::to_string(i));
		};
	auto summarize = [](YBWC_Node<std::string, int>& node) { return 0; };
	auto eval = [](std::string pos, OpenInterval) { return static_cast<int>(std::stoul(pos, nullptr, 16)); };
	ParallelTree<std::string, int> tree{ "0", { -1000000, 1000000 }, expand, summarize };

	#pragma omp parallel
	tree.work_on(eval);

	EXPECT_TRUE(tree.is_solved());
}