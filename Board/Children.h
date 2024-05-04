#pragma once
#include "Base/Base.h"
#include "Position.h"
#include "PossibleMoves.h"
#include <iterator>
#include <set>
#include <vector>

namespace children
{
	class Iterator
	{
		int plies = 0;
		bool pass_is_a_ply = false;
		std::vector<Position> pos_stack{};
		std::vector<Moves> moves_stack{};
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Position;
		using reference = Position&;
		using pointer = Position*;
		using iterator_category = std::forward_iterator_tag;

		Iterator() noexcept = default; // unused but required for std::forward_iterator.
		CUDA_CALLABLE Iterator(int plies, bool pass_is_a_ply) noexcept : plies(plies), pass_is_a_ply(pass_is_a_ply) {}
		CUDA_CALLABLE Iterator(const Position& start, int plies, bool pass_is_a_ply) noexcept;

		CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return (plies == o.plies) && (pass_is_a_ply == o.pass_is_a_ply) && (pos_stack == o.pos_stack) && (moves_stack == o.moves_stack); }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return !(*this == o); }

		CUDA_CALLABLE Iterator& operator++();
		CUDA_CALLABLE Iterator operator++(int) { Iterator tmp = *this; ++*this; return tmp; }
		CUDA_CALLABLE const Position& operator*() const noexcept { return pos_stack.back(); }
		CUDA_CALLABLE const Position* operator->() const noexcept { return &(pos_stack.back()); }
	};

	class Generator
	{
		Position pos;
		int plies;
		bool pass_is_a_ply;
	public:
		Generator(Position pos, int plies, bool pass_is_a_ply) noexcept
			: pos(pos), plies(plies), pass_is_a_ply(pass_is_a_ply) {}

		CUDA_CALLABLE Iterator begin() const { return { pos, plies, pass_is_a_ply }; }
		CUDA_CALLABLE Iterator end() const { return { plies, pass_is_a_ply }; }
	};
}

CUDA_CALLABLE children::Generator Children(Position, int plies, bool pass_is_a_ply);
CUDA_CALLABLE children::Generator Children(Position, int empty_count);

std::set<Position> UniqueChildren(Position, int empty_count);
