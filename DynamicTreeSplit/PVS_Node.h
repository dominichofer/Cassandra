#pragma once
#include "Base/Base.h"
#include "TreeNode.h"
#include <algorithm>
#include <cassert>
#include <optional>
#include <string>

enum class NodeType { PV, All, Cut };
enum class NodeStatus { Unsolved, Blocked, Solved };

std::string to_string(NodeType);
std::string to_string(NodeStatus);

inline std::string to_string(const std::string& value)
{
    return value;
}

template <typename Position, typename Result>
class YBWC_Node final : public TreeNode<YBWC_Node<Position, Result>>
{
    NodeStatus status = NodeStatus::Unsolved;
public:
    Position pos;
    OpenInterval window;
    Result result{};
    std::function<void()> stop_callback = [] {};

    YBWC_Node(Position pos, OpenInterval window) noexcept : pos(pos), window(window) {}

    bool is_unsolved() const { return status == NodeStatus::Unsolved; }
    bool is_blocked() const { return status == NodeStatus::Blocked; }
    bool is_solved() const { return status == NodeStatus::Solved; }

    // block and update parents
    void block(std::function<void(void)> stop_callback)
    {
        assert(this->is_leaf());
        assert(is_unsolved());
        this->status = NodeStatus::Blocked;
        this->stop_callback = std::move(stop_callback);
        if (not this->is_root())
            this->parent->update_status();
    }

    // stop and stop children
    void stop()
    {
        if (is_solved())
            return;
        status = NodeStatus::Unsolved;
        stop_callback();
        for (YBWC_Node& child : this->children)
            child.stop();
    }
    void solve(Result result, const std::function<Result(YBWC_Node&)>& summarize)
    {
        assert(is_unsolved() or is_blocked());
        stop();
        this->children.clear();
        this->result = std::move(result);
        status = NodeStatus::Solved;
        if (this->is_root())
            return;
        Result r = summarize(*this->parent);
        if (r > this->parent->window or this->parent->all_solved())
            this->parent->solve(r, summarize);
        else
            this->parent->update_status();
    }

    void append(Position pos)
    {
        TreeNode<YBWC_Node>::append(YBWC_Node(std::move(pos), -window));
    }

    std::string to_string() const
    {
        return std::format(
            "pos: {}, window: {}, status: {}, result: {}",
            pos,
            window,
            to_string(status),
            to_string(result)
        );
    }

    bool all_solved() const { return std::ranges::all_of(this->children, &YBWC_Node::is_solved); }
private:
    auto ParallelChildren() { return this->children | std::ranges::views::drop(1); }
    auto ParallelChildren() const { return this->children | std::ranges::views::drop(1); }

    bool no_parallel_unsolved() const { return std::ranges::none_of(ParallelChildren(), &YBWC_Node::is_unsolved); }

    bool children_block() const
    {
        assert(not this->is_leaf());
        auto& first = this->children.front();
        return first.is_blocked() or (first.is_solved() and no_parallel_unsolved());
    }

    void update_status()
    {
        assert(is_unsolved() or is_blocked());
        assert(not this->is_leaf());
        NodeStatus new_status = children_block() ? NodeStatus::Blocked : NodeStatus::Unsolved;
        if (status != new_status)
        {
            status = new_status;
            if (not this->is_root())
                this->parent->update_status();
        }
    }
};

template <typename Position, typename Result>
std::string to_string(const YBWC_Node<Position, Result>& node)
{
    int indent = 0;
    std::string ret;
    traverse_depth_first(
        node,
        [&](const YBWC_Node& node) { ret += std::string(indent++, ' ') + node.to_string() + '\n'; },
        [&](const YBWC_Node& node) { indent--; }
    );
    return ret;
}

template <typename Position, typename Result, typename Fkt>
YBWC_Node<Position, Result>* next_unsolved_leaf_node(
    YBWC_Node<Position, Result>& node,
    Fkt&& expand)
{
    if (not node.is_unsolved())
        return nullptr;
    if (node.is_leaf())
        expand(node);
    if (node.is_leaf())
        return std::addressof(node);

    for (auto& child : node.children)
        if (child.is_unsolved())
            return next_unsolved_leaf_node(child, expand);
    return nullptr;
}
