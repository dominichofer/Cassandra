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
