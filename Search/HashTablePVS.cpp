#include "HashTablePVS.h"
#include <algorithm>
#include <mutex>

void OneNode::Update(const key_type& new_key, const value_type& new_value)
{
	//auto correct_window = Search::AlphaBetaFailSoft{}.Eval(key).window;
	//if (!new_value.window.Contains(correct_window))
	//	return;
	std::scoped_lock lock{ mutex };
	//if (new_value.result.intensity < value.result.intensity)
	//	return;
	//if ((new_key == key) and (new_value.result.intensity < value.result.intensity))
	//	return;
	if (new_key == key)
	{
		if (new_value.result.intensity < value.result.intensity)
			return;
		value.result = new_value.result;
		//if (new_value.best_move == value.best_move_2)
		//	std::swap(value.best_move, value.best_move_2);
		//else if (new_value.best_move != value.best_move)
		//{
			//value.best_move_2 = value.best_move;
			value.best_move = new_value.best_move;
		//}
		//if (new_value.result.intensity == value.result.intensity)
		//	value.result.window = Overlap(value.result.window, new_value.result.window);
	}
	else
	{
	//if (!(new_value.result.intensity < value.result.intensity))
	//{
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
