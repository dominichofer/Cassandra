#pragma once
#include <ranges>
#include "range/v3/all.hpp"

template <typename T, typename ValueType>
concept range = std::ranges::range<T> and std::same_as<std::ranges::range_value_t<T>, ValueType>; // TODO: Rename to range_of

template <typename T, typename ValueType>
concept random_access_range = std::ranges::random_access_range<T> and std::same_as<std::ranges::range_value_t<T>, ValueType>; // TODO: Rename to random_access_range_of

template <typename T>
concept nested_range = std::ranges::range<T> and std::ranges::range<std::ranges::range_value_t<T>>;

template <typename T, typename ValueType>
concept nested_range_of = std::ranges::range<T> and (range<std::ranges::range_value_t<T>, ValueType> or nested_range_of<std::ranges::range_value_t<T>, ValueType>);