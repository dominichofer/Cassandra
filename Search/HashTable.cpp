#include "HashTable.h"
#include <algorithm>
#include <mutex>

void OneNode::Update(const key_type& new_key, const value_type& new_value)
{
	std::scoped_lock lock{ mutex };
	if (new_value.depth < value.depth or (new_value.depth == value.depth and new_value.confidence_level < value.confidence_level))
		return;

	if (key == new_key and new_value.depth == value.depth and new_value.confidence_level == value.confidence_level)
	{
		value.best_move = new_value.best_move;
		value.window = Intersection(value.window, new_value.window);
		if (value.window.lower > value.window.upper)
			value.window = { min_score, max_score };
	}
	else
	{
		key = new_key;
		value = new_value;
	}
}

std::optional<OneNode::value_type> OneNode::LookUp(const key_type& key) const
{
	std::scoped_lock lock{ mutex };
	if (key == this->key)
		return value;
	return std::nullopt;
}

void OneNode::Clear()
{
	std::scoped_lock lock{ mutex };
	key = {};
	value = {};
}

void TwoNodes::Update(const key_type& new_key, const value_type& new_value)
{
	std::scoped_lock lock{ mutex };
	if (new_value.depth >= value1.depth and new_value.confidence_level >= value1.confidence_level)
	{
		key1 = new_key;
		value1 = new_value;
	}
	else if (new_value.depth >= value2.depth and new_value.confidence_level >= value2.confidence_level)
	{
		key2 = new_key;
		value2 = new_value;
	}
}

std::optional<TwoNodes::value_type> TwoNodes::LookUp(const key_type& key) const
{
	std::scoped_lock lock{ mutex };
	if (key == this->key1)
		return value1;
	if (key == this->key2)
		return value2;
	return std::nullopt;
}

void TwoNodes::Clear()
{
	std::scoped_lock lock{ mutex };
	key1 = {};
	key2 = {};
	value1 = DefaultValue();
	value2 = DefaultValue();
}
