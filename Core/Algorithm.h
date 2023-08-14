#pragma once
#include <functional>
#include <random>
#include <ranges>
#include <string>
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

template <typename Iterable, typename Projection = std::identity>
std::string join(const std::string& separator, const Iterable& iterable, Projection proj = {})
{
	if (std::empty(iterable))
		return {};

	std::string ret = proj(*std::begin(iterable));
	for (auto it = std::begin(iterable) + 1; it != std::end(iterable); ++it)
		ret += separator + proj(*it);
	return ret;
}


template <typename Iterable, typename Projection = std::identity>
std::string join(char separator, const Iterable& iterable, Projection proj = {})
{
	return join(std::string{separator}, iterable, proj);
}