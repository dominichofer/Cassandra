#pragma once
#include <ranges>
#include "range/v3/all.hpp"

template <typename T, typename ValueType>
concept range = std::ranges::range<T> and std::same_as<std::ranges::range_value_t<T>, ValueType>;
