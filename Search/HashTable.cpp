#include "HashTable.h"
#include <algorithm>
#include <mutex>

void OneNode::Update(const key_type& new_key, const value_type& new_value)
{
	std::scoped_lock lock{ mutex };
	if (new_value.intensity >= value.intensity)
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
	value = DefaultValue();
}

void TwoNodes::Update(const key_type& new_key, const value_type& new_value)
{
	std::scoped_lock lock{ mutex };
	if (new_value.intensity >= value1.intensity)
	{
		key1 = new_key;
		value1 = new_value;
	}
	else if (new_value.intensity >= value2.intensity)
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
