#pragma once
#include "Core/Core.h"
#include <iterator>

int pown(int base, unsigned int exponent);

int FastIndex(Position, uint64_t pattern) noexcept;

class Configurations
{
	uint64_t mask;
public:
	class Iterator
	{
		uint64_t mask{};
		uint64_t size{0}, p{0}, o{0};
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		Iterator() noexcept = default;
		Iterator(uint64_t mask) noexcept;

		Iterator& operator++();
		Position operator*() const;

		bool operator==(const Iterator&) const noexcept = default;
		bool operator!=(const Iterator&) const noexcept = default;
	};

	Configurations(uint64_t mask) noexcept : mask(mask) {}

	Iterator begin() const noexcept { return mask; }
	Iterator cbegin() const noexcept { return mask; }
	static Iterator end() noexcept { return {}; }
	static Iterator cend() noexcept { return {}; }
};
