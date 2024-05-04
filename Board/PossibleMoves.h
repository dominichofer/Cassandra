#pragma once
#include "Base/Base.h"
#include "Field.h"
#include "Position.h"
#include <iterator>

namespace moves
{
	class Iterator
	{
		uint64_t moves;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		CUDA_CALLABLE Iterator() noexcept : moves(0) {}
		CUDA_CALLABLE Iterator(const uint64_t& moves) noexcept : moves(moves) {}

		CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return moves == o.moves; }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return moves != o.moves; }

		CUDA_CALLABLE Iterator& operator++() noexcept { ClearLSB(moves); return *this; }
		CUDA_CALLABLE Field operator*() const noexcept { return static_cast<Field>(std::countr_zero(moves)); }
	};
}

class Moves
{
	uint64_t b{ 0 };
public:
	Moves() noexcept = default;
	CUDA_CALLABLE explicit Moves(uint64_t moves) noexcept : b(moves) {}

	CUDA_CALLABLE bool operator==(const Moves& o) const noexcept { return b == o.b; }
	CUDA_CALLABLE bool operator!=(const Moves& o) const noexcept { return b != o.b; }
	CUDA_CALLABLE Moves operator&(const uint64_t& mask) const noexcept { return Moves{ b & mask }; }

	CUDA_CALLABLE Field operator[](std::size_t index) const noexcept { return static_cast<Field>(std::countr_zero(PDep(1ULL << index, b))); }

	CUDA_CALLABLE operator uint64_t() const noexcept { return b; }

	CUDA_CALLABLE bool empty() const noexcept { return b == 0; }
	CUDA_CALLABLE bool contains(Field move) const noexcept { return b & Bit(move); }
	CUDA_CALLABLE void erase(Field move) noexcept { b &= ~Bit(move); }
	CUDA_CALLABLE std::size_t size() const noexcept { return std::popcount(b); }

	CUDA_CALLABLE Field front() const noexcept { return static_cast<Field>(std::countr_zero(b)); }
	CUDA_CALLABLE void pop_front() noexcept { ClearLSB(b); }

	CUDA_CALLABLE moves::Iterator begin() const noexcept { return b; }
	CUDA_CALLABLE moves::Iterator end() const noexcept { return {}; }
};

CUDA_CALLABLE Moves PossibleMoves(const Position&) noexcept;

namespace detail
{
	#ifdef __AVX512F__
		Moves PossibleMoves_AVX512(const Position&) noexcept;
	#endif

	#ifdef __AVX2__
		Moves PossibleMoves_AVX2(const Position&) noexcept;
	#endif

	CUDA_CALLABLE Moves PossibleMoves_x64(const Position&) noexcept;
}
