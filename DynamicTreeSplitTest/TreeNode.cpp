#include "pch.h"
#include <string>

class SimpleNode : public TreeNode<SimpleNode>
{
public:
    char name;

    SimpleNode(char name) noexcept : name(name) {}
};

SimpleNode SimpleTree()
{
    SimpleNode a{ 'a' };
    auto& b = a.append({ 'b' });
    b.append({ 'c' });
    b.append({ 'd' });
    b.append({ 'e' });
    auto& f = a.append({ 'f' });
    f.append({ 'g' });
    f.append({ 'h' });
    return a;
}

TEST(TreeNode, traverse_depth_first)
{
    auto root = SimpleTree();

    std::string children;
    auto append = [&children](const SimpleNode& node)
        { children.push_back(node.name); };
    auto append_upper = [&children](const SimpleNode& node)
        { children.push_back(std::toupper(node.name)); };

    traverse_depth_first(root, append, append_upper);

    EXPECT_EQ(children, "abcCdDeEBfgGhHFA");
}

TEST(TreeNode, Descendant)
{
    auto root = SimpleTree();

    EXPECT_EQ(root.descendant({})->name, 'a');
    EXPECT_EQ(root.descendant({ 0 })->name, 'b');
    EXPECT_EQ(root.descendant({ 0, 0 })->name, 'c');
    EXPECT_EQ(root.descendant({ 0, 1 })->name, 'd');
    EXPECT_EQ(root.descendant({ 0, 2 })->name, 'e');
    EXPECT_EQ(root.descendant({ 1 })->name, 'f');
    EXPECT_EQ(root.descendant({ 1, 0 })->name, 'g');
    EXPECT_EQ(root.descendant({ 1, 1 })->name, 'h');
    EXPECT_EQ(root.descendant({ 1, 2 }), nullptr);
}
