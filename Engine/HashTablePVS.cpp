#include "HashTablePVS.h"
#include <algorithm>
#include <mutex>

OneNode::OneNode(const OneNode& o) noexcept
{
	std::lock_guard<SpinlockMutex> o_lock{ o.mutex };
	node = o.node;
}

OneNode& OneNode::operator=(const OneNode& o) noexcept
{
	if (&o != this)
		return *this;
	std::lock_guard<SpinlockMutex> my_lock{ mutex };
	std::lock_guard<SpinlockMutex> o_lock{ o.mutex };
	node = o.node;
	return *this;
}

void OneNode::Update(Position key, const PVS_Info& novum)
{
	std::lock_guard<SpinlockMutex> lock{ mutex };
	if (key != node.key)
	{
		node = { key, novum };
		return;
	}
	
	const bool novum_is_exact = (novum.depth == key.EmptyCount()) && (novum.selectivity == Search::Selectivity::None);
	const bool node_is_exact = (node.value.depth == key.EmptyCount()) && (node.value.selectivity == Search::Selectivity::None);
	if (novum_is_exact && node_is_exact)
	{
		assert(std::max(node.value.window.lower, novum.window.lower) <= std::min(node.value.window.upper, novum.window.upper));

		node.value.node_count += novum.node_count;

		if (novum.window.lower > node.value.window.lower)
		{
			node.value.window.lower = novum.window.lower;
			node.value.best_move = novum.best_move;
		}
		if (novum.window.upper < node.value.window.upper)
		{
			node.value.window.upper = novum.window.upper;
			node.value.best_move = novum.best_move;
		}
	}
	else
		node.value = novum;
}

std::optional<PVS_Info> OneNode::LookUp(Position key) const
{
	std::lock_guard<SpinlockMutex> lock{ mutex };
	if (key == node.key)
		return node.value;
	return {};
}

void OneNode::Clear()
{
	std::lock_guard<SpinlockMutex> lock{ mutex };
	node = Node();
}

std::size_t OneNode::NumberOfNonEmptyNodes() const
{
	return (node.key == Node{}.key) ? 0 : 1;
}
