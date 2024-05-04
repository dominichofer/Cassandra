#include "pch.h"
#include <string>
#include <ranges>
#include <algorithm>

TEST(YBWC_Node, leaf_is_blocked_after_block)
{
    YBWC_Node<char, int> node{ ' ', { -2, +2 } };
    node.block([] {});
    EXPECT_TRUE(node.is_blocked());
}

TEST(YBWC_Node, block_next_unsolved_leaf_node)
{
    //     a
    //    / \
    //   b   c
    //  / \
    // d   e
    YBWC_Node<char, int> root{ 'a', { 0, +1 } };
    root.append('b');
    root.append('c');
    root.children[0].append('d');
    root.children[0].append('e');

    //     (0,+1) -> 1
    //     /    \ <- cut
    //    -1     0
    //   / \
    //  1   1

    auto expand = [](YBWC_Node<char, int>&) {};
    auto summarize = [](YBWC_Node<char, int>& node) {
        auto solveds = std::ranges::views::filter(node.children, &YBWC_Node<char, int>::is_solved);
        auto results = std::ranges::views::transform(solveds, &YBWC_Node<char, int>::result);
        return -std::ranges::min(results);
        };

    // blocking d, blocks root
    auto next = next_unsolved_leaf_node(root, expand);
    next->block([] {});
    EXPECT_EQ(next->pos, 'd');
    EXPECT_EQ(next_unsolved_leaf_node(root, expand), nullptr);
    next->solve(1, summarize);

    // blocking e, blocks root
    next = next_unsolved_leaf_node(root, expand);
    next->block([] {});
    EXPECT_EQ(next->pos, 'e');
    EXPECT_EQ(next_unsolved_leaf_node(root, expand), nullptr);
    next->solve(1, summarize);

    //// root blocked
    //next = next_unsolved_leaf_node(root, expand);
    //next->block([] {});
    //EXPECT_EQ(next->pos, 'g');

    //// solve g -> beta cut
    //next->solve(0, summarize);

    //// block d
    //next = next_unsolved_leaf_node(root, expand);
    //next->block([] {});
    //EXPECT_EQ(next->pos, 'd');

    //// solve d
    //next->solve(0, summarize);

    EXPECT_EQ(root.result, 1);

}

//TEST(YBWC_Node, nibling_cut)
//{
//    //    r
//    //  /   \
//    // a     b
//    //      / \ <- cut 
//    //     c   d
//    // 
//    //  (-3,+3) -> -1
//    //  /    \
//    // 1      b
//    //       / \ <- cut 
//    //      -2  d
//
//    YBWC_Node<char, int> r{ 'r', { -3, +3 } };
//    r.append('a');
//    r.append('b');
//    r.children[1].append('c');
//    r.children[1].append('d');
//
//    auto expand = [](YBWC_Node<char, int>&) {};
//    auto summarize = [](YBWC_Node<char, int>& node) {
//        auto solveds = std::ranges::views::filter(node.children, &YBWC_Node<char, int>::is_solved);
//        auto results = std::ranges::views::transform(solveds, &YBWC_Node<char, int>::result);
//        return -std::ranges::min(results);
//        };
//
//    // block a
//    auto a = next_unsolved_leaf_node(r, expand);
//    a->block([] {});
//    EXPECT_EQ(a->pos, 'a');
//
//    // block c
//    auto c = next_unsolved_leaf_node(r, expand);
//    c->block([] {});
//    EXPECT_EQ(c->pos, 'c');
//
//    // solve c
//    c->solve(-2, summarize);
//
//    // solve a -> cut d
//    a->solve(+1, summarize);
//
//    EXPECT_EQ(r.result, -1);
//}