#pragma once
#include "Position.h"
#include <iterator>

class ChildrenGenerator
{
	class Iterator
	{
		struct PosMov
		{
			Position pos;
			Moves moves;

			CUDA_CALLABLE PosMov(Position pos, Moves moves) : pos(pos), moves(moves) {}
			CUDA_CALLABLE bool operator==(const PosMov& o) const noexcept { return std::tie(pos, moves) == std::tie(o.pos, o.moves); }
			CUDA_CALLABLE bool operator!=(const PosMov& o) const noexcept { return std::tie(pos, moves) != std::tie(o.pos, o.moves); }
		};
		int plies = 0;
		bool pass_is_a_ply = false;
		std::vector<PosMov> stack{};
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Field;
		using reference = Field&;
		using pointer = Field*;
		using iterator_category = std::forward_iterator_tag;

		CUDA_CALLABLE Iterator() noexcept = default;
		CUDA_CALLABLE Iterator(const Position& start, int plies, bool pass_is_a_ply) noexcept;

		CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return (stack.empty() && o.stack.empty()) || std::tie(plies, pass_is_a_ply, stack) == std::tie(o.plies, o.pass_is_a_ply, o.stack); }
		CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return !(*this == o); }
		CUDA_CALLABLE Iterator& operator++();
		CUDA_CALLABLE const Position& operator*() const noexcept { return stack.back().pos; }
		CUDA_CALLABLE const Position* operator->() const noexcept { return std::addressof(stack.back().pos); }
	};

	const Position start;
	const int plies;
	const bool pass_is_a_ply;
public:
	ChildrenGenerator(const Position& start, int plies, bool pass_is_a_ply) noexcept
		: start(start), plies(plies), pass_is_a_ply(pass_is_a_ply) {}

	CUDA_CALLABLE Iterator begin() const { return { start, plies, pass_is_a_ply }; }
	CUDA_CALLABLE Iterator cbegin() const { return { start, plies, pass_is_a_ply }; }
	CUDA_CALLABLE static Iterator end() { return {}; }
	CUDA_CALLABLE static Iterator cend() { return {}; }
};

CUDA_CALLABLE ChildrenGenerator Children(Position, int plies, bool pass_is_a_ply);
CUDA_CALLABLE ChildrenGenerator Children(Position, int empty_count);
