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
	, m_half_size(Pow_int(3, PopCount(pattern & HALF)))
{
	assert(pattern == FlipHorizontal(pattern));
}

std::vector<BitBoard> HorizontalSymmetric::Patterns() const
{
	return { Pattern, m_pattern_C, m_pattern_V, m_pattern_D };
}

std::vector<int> HorizontalSymmetric::ReducedIndices(Position pos) const
{
	return { ReducedIndex0(pos), ReducedIndex1(pos), ReducedIndex2(pos), ReducedIndex3(pos) };
}

std::size_t HorizontalSymmetric::ReducedSize() const
{
	return m_half_size * (m_half_size + 1) / 2;
}

int HorizontalSymmetric::ReducedIndex0(Position pos) const
{
	int min = ReducedIndex(pos, Pattern & HALF);
	pos.FlipHorizontal();
	int max = ReducedIndex(pos, Pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return min * m_half_size + max - (min * (min + 1) / 2);
}

int HorizontalSymmetric::ReducedIndex1(Position pos) const
{
	pos.FlipCodiagonal();
	return ReducedIndex0(pos);
}

int HorizontalSymmetric::ReducedIndex2(Position pos) const
{
	pos.FlipVertical();
	return ReducedIndex0(pos);
}

int HorizontalSymmetric::ReducedIndex3(Position pos) const
{
	pos.FlipDiagonal();
	return ReducedIndex0(pos);
}

DiagonalSymmetric::DiagonalSymmetric(BitBoard pattern)
	: IndexMapper(pattern)
	, m_pattern_H(FlipHorizontal(pattern))
	, m_pattern_C(FlipCodiagonal(pattern))
	, m_pattern_V(FlipVertical(pattern))
	, m_half_size(Pow_int(3, PopCount(pattern & HALF)))
	, m_diag_size(Pow_int(3, PopCount(pattern & DIAG)))
{
	assert(pattern == FlipDiagonal(pattern));
}

std::vector<BitBoard> DiagonalSymmetric::Patterns() const
{
	return { Pattern, m_pattern_H, m_pattern_C, m_pattern_V };
}

std::vector<int> DiagonalSymmetric::ReducedIndices(Position pos) const
{
	return { ReducedIndex0(pos), ReducedIndex1(pos), ReducedIndex2(pos), ReducedIndex3(pos) };
}

std::size_t DiagonalSymmetric::ReducedSize() const
{
	return m_diag_size * m_half_size * (m_half_size + 1) / 2;
}

int DiagonalSymmetric::ReducedIndex0(Position pos) const
{
	int diag = ReducedIndex(pos, Pattern & DIAG);

	int min = ReducedIndex(pos, Pattern & HALF);
	pos.FlipDiagonal();
	int max = ReducedIndex(pos, Pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return (min * m_half_size + max - (min * (min + 1) / 2)) * m_diag_size + diag;
}

int DiagonalSymmetric::ReducedIndex1(Position pos) const
{
	pos.FlipHorizontal();
	return ReducedIndex0(pos);
}

int DiagonalSymmetric::ReducedIndex2(Position pos) const
{
	pos.FlipCodiagonal();
	return ReducedIndex0(pos);
}

int DiagonalSymmetric::ReducedIndex3(Position pos) const
{
	pos.FlipVertical();
	return ReducedIndex0(pos);
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

std::vector<int> Asymmetric::ReducedIndices(Position pos) const
{
	return {
		ReducedIndex0(pos), ReducedIndex1(pos), ReducedIndex2(pos), ReducedIndex3(pos),
		ReducedIndex4(pos), ReducedIndex5(pos), ReducedIndex6(pos), ReducedIndex7(pos)
	};
}

std::size_t Asymmetric::ReducedSize() const
{
	return Pow_int(3, PopCount(Pattern));
}

int Asymmetric::ReducedIndex0(Position pos) const
{
	return ReducedIndex(pos, Pattern);
}

int Asymmetric::ReducedIndex1(Position pos) const
{
	pos.FlipHorizontal();
	return ReducedIndex0(pos);
}

int Asymmetric::ReducedIndex2(Position pos) const
{
	pos.FlipVertical();
	return ReducedIndex0(pos);
}

int Asymmetric::ReducedIndex3(Position pos) const
{
	pos.FlipDiagonal();
	return ReducedIndex0(pos);
}

int Asymmetric::ReducedIndex4(Position pos) const
{
	pos.FlipCodiagonal();
	return ReducedIndex0(pos);
}

int Asymmetric::ReducedIndex5(Position pos) const
{
	pos.FlipVertical();
	pos.FlipHorizontal();
	return ReducedIndex0(pos);
}

int Asymmetric::ReducedIndex6(Position pos) const
{
	pos.FlipDiagonal();
	pos.FlipHorizontal();
	return ReducedIndex0(pos);
}

int Asymmetric::ReducedIndex7(Position pos) const
{
	pos.FlipCodiagonal();
	pos.FlipHorizontal();
	return ReducedIndex0(pos);
}
