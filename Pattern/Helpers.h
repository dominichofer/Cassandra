#pragma once
#include "Core/Position.h"
#include <cstdint>
#include <functional>

uint64_t Pow_int(uint64_t base, uint64_t exponent);

int Index(const Position&, BitBoard pattern) noexcept;

// Configuration generator
class Configurations
{
	BitBoard pattern;

	class Iterator
	{
		BitBoard pattern{};
		uint64_t size{0}, p{0}, o{0};
	public:
		Iterator() noexcept = default;
		Iterator(BitBoard pattern) noexcept : pattern(pattern), size(1ULL << popcount(pattern)) {}

		[[nodiscard]] Iterator& operator++() {
			o++;
			for (; p < size; p++) {
				for (; o < size; o++)
					if ((p & o) == 0u) // fields only be taken by one player.
						return *this;
				o = 0;
			}
			*this = Iterator{}; // marks generator as depleted.
			return *this;
		}
		[[nodiscard]] Position operator*() const { return Position::From(PDep(p, pattern), PDep(o, pattern)); }

		[[nodiscard]] bool operator==(const Iterator& o) const noexcept = default;
		[[nodiscard]] bool operator!=(const Iterator& o) const noexcept = default;
	};
public:
	Configurations(BitBoard pattern) noexcept : pattern(pattern) {}

	[[nodiscard]] Iterator begin() const { return Iterator(pattern); }
	[[nodiscard]] Iterator cbegin() const { return Iterator(pattern); }
	[[nodiscard]] Iterator end() const { return {}; }
	[[nodiscard]] Iterator cend() const { return {}; }
};
