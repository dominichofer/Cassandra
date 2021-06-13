#pragma once

template <typename InIt, typename T, typename F>
[[nodiscard]]
T sum(InIt first, const InIt last, T value, F transform)
{
	for (; first != last; ++first)
		value += transform(*first);
	return value;
}

template <typename InIt, typename T>
[[nodiscard]]
T sum(InIt first, const InIt last, T value)
{
	for (; first != last; ++first)
		value += *first;
	return value;
}