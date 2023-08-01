#pragma once
#include <random>
#include <ranges>
#include <vector>

template <std::ranges::input_range R>
auto Sample(int size, R&& pool, uint64_t seed = std::random_device{}())
{
	std::mt19937_64 rnd_engine(seed);

	std::vector<std::ranges::range_value_t<R>> samples;
	samples.reserve(size);
	std::ranges::sample(
		pool,
		std::back_inserter(samples),
		size, rnd_engine);
	return samples;
}

template <class Iterable>
std::string join(const std::string& delimiter, const Iterable& iterable)
{
    using std::to_string;

    if (std::empty(iterable))
        return {};

    std::string ret = to_string(*std::begin(iterable));
    for (auto it = std::begin(iterable) + 1; it != std::end(iterable); ++it)
        ret += delimiter + to_string(*it);
    return ret;
}

template <class Iterable>
std::string join(char delimiter, const Iterable& iterable)
{
    return join(std::string{delimiter}, iterable);
}
