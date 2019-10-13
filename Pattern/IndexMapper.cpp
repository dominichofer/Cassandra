#include "IndexMapper.h"
#include "Helpers.h"
#include <cassert>

std::size_t IndexMapper::Multiplicity() const
{
	return std::size(Patterns());
}

std::unique_ptr<IndexMapper> CreateIndexMapper(const BitBoard pattern)
{
	if (pattern == FlipHorizontal(pattern))
		return std::make_unique<HorizontalSymmetric>(pattern);
	if (pattern == FlipDiagonal(pattern))
		return std::make_unique<DiagonalSymmetric>(pattern);
	return std::make_unique<Asymmetric>(pattern);
}

HorizontalSymmetric::HorizontalSymmetric(BitBoard pattern)
	: IndexMapper(pattern)
	, m_pattern_C(FlipCodiagonal(pattern))
	, m_pattern_V(FlipVertical(pattern))
	, m_pattern_D(FlipDiagonal(pattern))
	, m_half_size(Pow_int(3, PopCount(pattern & HALF & ~BitBoard::Middle())) * Pow_int(2, PopCount(pattern & HALF & BitBoard::Middle())))
{
	assert(pattern == FlipHorizontal(pattern));
}

std::vector<BitBoard> HorizontalSymmetric::Patterns() const
{
	return { Pattern, m_pattern_C, m_pattern_V, m_pattern_D };
}

std::vector<int> HorizontalSymmetric::ReducedIndices(Board board) const
{
	return { ReducedIndex0(board), ReducedIndex1(board), ReducedIndex2(board), ReducedIndex3(board) };
}

std::size_t HorizontalSymmetric::ReducedSize() const
{
	return m_half_size * (m_half_size + 1) / 2;
}

int HorizontalSymmetric::ReducedIndex0(Board board) const
{
	int min = ReducedIndex(board, Pattern & HALF);
	board.FlipHorizontal();
	int max = ReducedIndex(board, Pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return min * m_half_size + max - (min * (min + 1) / 2);
}

int HorizontalSymmetric::ReducedIndex1(Board board) const
{
	board.FlipCodiagonal();
	return ReducedIndex0(board);
}

int HorizontalSymmetric::ReducedIndex2(Board board) const
{
	board.FlipVertical();
	return ReducedIndex0(board);
}

int HorizontalSymmetric::ReducedIndex3(Board board) const
{
	board.FlipDiagonal();
	return ReducedIndex0(board);
}

DiagonalSymmetric::DiagonalSymmetric(BitBoard pattern)
	: IndexMapper(pattern)
	, m_pattern_H(FlipHorizontal(pattern))
	, m_pattern_C(FlipCodiagonal(pattern))
	, m_pattern_V(FlipVertical(pattern))
	, m_half_size(Pow_int(3, PopCount(pattern & HALF & ~BitBoard::Middle())) * Pow_int(2, PopCount(pattern & HALF & BitBoard::Middle())))
	, m_diag_size(Pow_int(3, PopCount(pattern & DIAG & ~BitBoard::Middle())) * Pow_int(2, PopCount(pattern & DIAG & BitBoard::Middle())))
{
	assert(pattern == FlipDiagonal(pattern));
}

std::vector<BitBoard> DiagonalSymmetric::Patterns() const
{
	return { Pattern, m_pattern_H, m_pattern_C, m_pattern_V };
}

std::vector<int> DiagonalSymmetric::ReducedIndices(Board board) const
{
	return { ReducedIndex0(board), ReducedIndex1(board), ReducedIndex2(board), ReducedIndex3(board) };
}

std::size_t DiagonalSymmetric::ReducedSize() const
{
	return m_diag_size * m_half_size * (m_half_size + 1) / 2;
}

int DiagonalSymmetric::ReducedIndex0(Board board) const
{
	int diag = ReducedIndex(board, Pattern & DIAG);

	int min = ReducedIndex(board, Pattern & HALF);
	board.FlipDiagonal();
	int max = ReducedIndex(board, Pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return (min * m_half_size + max - (min * (min + 1) / 2)) * m_diag_size + diag;
}

int DiagonalSymmetric::ReducedIndex1(Board board) const
{
	board.FlipHorizontal();
	return ReducedIndex0(board);
}

int DiagonalSymmetric::ReducedIndex2(Board board) const
{
	board.FlipCodiagonal();
	return ReducedIndex0(board);
}

int DiagonalSymmetric::ReducedIndex3(Board board) const
{
	board.FlipVertical();
	return ReducedIndex0(board);
}

Asymmetric::Asymmetric(BitBoard pattern)
	: IndexMapper(pattern)
	, m_pattern_H(FlipHorizontal(pattern))
	, m_pattern_V(FlipVertical(pattern))
	, m_pattern_D(FlipDiagonal(pattern))
	, m_pattern_C(FlipCodiagonal(pattern))
	, m_patternHV(FlipVertical(FlipHorizontal(pattern)))
	, m_patternHD(FlipDiagonal(FlipHorizontal(pattern)))
	, m_patternHC(FlipCodiagonal(FlipHorizontal(pattern)))
{}

std::vector<BitBoard> Asymmetric::Patterns() const
{
	return { Pattern, m_pattern_H, m_pattern_V, m_pattern_D, m_pattern_C, m_patternHV, m_patternHD, m_patternHC };
}

std::vector<int> Asymmetric::ReducedIndices(Board board) const
{
	return {
		ReducedIndex0(board), ReducedIndex1(board), ReducedIndex2(board), ReducedIndex3(board),
		ReducedIndex4(board), ReducedIndex5(board), ReducedIndex6(board), ReducedIndex7(board)
	};
}

std::size_t Asymmetric::ReducedSize() const
{
	return Pow_int(3, PopCount(Pattern & ~BitBoard::Middle())) * Pow_int(2, PopCount(Pattern & BitBoard::Middle()));
}

int Asymmetric::ReducedIndex0(Board board) const
{
	return ReducedIndex(board, Pattern);
}

int Asymmetric::ReducedIndex1(Board board) const
{
	board.FlipHorizontal();
	return ReducedIndex0(board);
}

int Asymmetric::ReducedIndex2(Board board) const
{
	board.FlipVertical();
	return ReducedIndex0(board);
}

int Asymmetric::ReducedIndex3(Board board) const
{
	board.FlipDiagonal();
	return ReducedIndex0(board);
}

int Asymmetric::ReducedIndex4(Board board) const
{
	board.FlipCodiagonal();
	return ReducedIndex0(board);
}

int Asymmetric::ReducedIndex5(Board board) const
{
	board.FlipVertical();
	board.FlipHorizontal();
	return ReducedIndex0(board);
}

int Asymmetric::ReducedIndex6(Board board) const
{
	board.FlipDiagonal();
	board.FlipHorizontal();
	return ReducedIndex0(board);
}

int Asymmetric::ReducedIndex7(Board board) const
{
	board.FlipCodiagonal();
	board.FlipHorizontal();
	return ReducedIndex0(board);
}
