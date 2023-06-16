#pragma once
#include "Position.h"
#include <iterator>
#include <set>
#include <vector>

namespace children
{
	struct PositionMoves
	{
		Position pos;
		Moves moves;

		CUDA_CALLABLE bool operator==(const PositionMoves& o) const noexcept { return pos == o.pos and moves == o.moves; }
		CUDA_CALLABLE bool operator!=(const PositionMoves& o) const noexcept { return not (*this == o); }
	};

	class Iterator
	{
		int plies = 0;
		bool pass_is_a_ply = false;
		std::vector<PositionMoves> stack{};
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Position;
		using reference = Position&;
		using pointer = Position*;
		using iterator_category = std::forward_iterator_tag;

		CUDA_CALLABLE Iterator() = default;
		CUDA_CALLABLE Iterator(int plies, bool pass_is_a_ply) noexcept : plies(plies), pass_is_a_ply(pass_is_a_ply) {}
		CUDA_CALLABLE Iterator(const Position& start, int plies, bool pass_is_a_ply) noexcept;

		CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return plies == o.plies and pass_is_a_ply == o.pass_is_a_ply and stack == o.stack; }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return not (*this == o); }

		CUDA_CALLABLE Iterator& operator++();
		CUDA_CALLABLE Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
		CUDA_CALLABLE const Position& operator*() const noexcept { return stack.back().pos; }
		CUDA_CALLABLE const Position* operator->() const noexcept { return std::addressof(stack.back().pos); }
	};

	class Generator
	{
		Position start;
		int plies;
		bool pass_is_a_ply;
	public:
		Generator(const Position& start, int plies, bool pass_is_a_ply) noexcept
			: start(start), plies(plies), pass_is_a_ply(pass_is_a_ply) {}

		CUDA_CALLABLE Iterator begin() const { return { start, plies, pass_is_a_ply }; }
		CUDA_CALLABLE Iterator end() const { return { plies, pass_is_a_ply }; }
	};
}

CUDA_CALLABLE children::Generator Children(Position, int plies, bool pass_is_a_ply);
CUDA_CALLABLE children::Generator Children(Position, int empty_count);

std::set<Position> UniqueChildren(Position, int empty_count);
