#pragma once
#include "Core/Position.h"
#include <functional>
#include <iterator>

template <typename T>
static constexpr inline T pown(T base, unsigned int exponent)
{
	T result = 1;
	while (exponent)
	{
		if (exponent % 2)
			result *= base;
		base *= base;
		exponent >>= 1;
	}
	return result;
}

int FastIndex(const Position&, BitBoard pattern) noexcept;

class Configurations
{
	BitBoard pattern;

	class Iterator
	{
		BitBoard pattern{};
		uint64_t size{0}, p{0}, o{0};
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		Iterator() noexcept = default;
		Iterator(BitBoard pattern) noexcept : pattern(pattern), size(1ULL << popcount(pattern)) {}

		Iterator& operator++() {
			o++;
			for (; p < size; p++) {
				for (; o < size; o++)
					if ((p & o) == 0u) // fields can only be taken by one player.
						return *this;
				o = 0;
			}
			*this = end(); // marks generator as depleted.
			return *this;
		}
		[[nodiscard]] Position operator*() const { return Position::From(PDep(p, pattern), PDep(o, pattern)); }

		[[nodiscard]] bool operator==(const Iterator&) const noexcept = default;
		[[nodiscard]] bool operator!=(const Iterator&) const noexcept = default;
	};
public:
	Configurations(BitBoard pattern) noexcept : pattern(pattern) {}

	[[nodiscard]] Iterator begin() const noexcept { return pattern; }
	[[nodiscard]] Iterator cbegin() const noexcept { return pattern; }
	[[nodiscard]] static Iterator end() noexcept { return {}; }
	[[nodiscard]] static Iterator cend() noexcept { return {}; }
};
