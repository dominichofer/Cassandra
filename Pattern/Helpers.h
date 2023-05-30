#pragma once
#include "Core/Core.h"
#include <functional>
#include <iterator>

int pown(int base, unsigned int exponent);

int FastIndex(Position, BitBoard pattern) noexcept;

class Configurations
{
	BitBoard pattern;
public:
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
		Position operator*() const { return { PDep(p, pattern), PDep(o, pattern) }; }

		bool operator==(const Iterator&) const noexcept = default;
		bool operator!=(const Iterator&) const noexcept = default;
	};

	Configurations(BitBoard pattern) noexcept : pattern(pattern) {}

	Iterator begin() const noexcept { return pattern; }
	Iterator cbegin() const noexcept { return pattern; }
	static Iterator end() noexcept { return {}; }
	static Iterator cend() noexcept { return {}; }
};
