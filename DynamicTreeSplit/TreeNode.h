#pragma once
#include <memory>
#include <optional>
#include <vector>
#include <functional>

template <typename Derived>
class TreeNode
{
protected:
    Derived* parent = nullptr;
public:
    std::vector<Derived> children;

    TreeNode() noexcept = default;
    //explicit TreeNode(std::vector<Derived> children, TreeNode<Derived>* parent) noexcept : children(std::move(children)), parent(parent) {}

    bool is_root() const { return parent == nullptr; }
    bool is_leaf() const { return children.empty(); }

    Derived& append(Derived child)
    {
        child.parent = static_cast<Derived*>(this);
        children.push_back(std::move(child));
        return children.back();
    }

    Derived* descendant(const std::vector<std::size_t>& index)
    {
        Derived* node = static_cast<Derived*>(this);
        for (std::size_t i : index)
            if (i < node->children.size())
                node = std::addressof(node->children[i]);
            else
                return nullptr;
        return node;
    }
};

template <typename Derived>
void traverse_depth_first(Derived& node, auto pre_children_op, auto post_children_op)
{
    pre_children_op(node);
    for (const Derived& child : node.children)
        traverse_depth_first(child, pre_children_op, post_children_op);
    post_children_op(node);
}