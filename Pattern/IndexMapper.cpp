#include "IndexMapper.h"
#include <cassert>
#include <numeric>

std::size_t IndexMapper::GroupOrder() const
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

std::unique_ptr<IndexMapper> CreateIndexMapper(const std::vector<BitBoard>& patterns)
{
	return std::make_unique<Composite>(patterns);
}

HorizontalSymmetric::HorizontalSymmetric(BitBoard pattern)
	: m_pattern(pattern)
	, m_pattern_C(FlipCodiagonal(pattern))
	, m_pattern_V(FlipVertical(pattern))
	, m_pattern_D(FlipDiagonal(pattern))
	, m_half_size(Pow_int(3, PopCount(pattern & HALF)))
{
	assert(pattern == FlipHorizontal(pattern));
}

void HorizontalSymmetric::PushBackIndices(std::vector<int>& vec, const Position& pos, std::size_t offset) const
{
	vec.push_back(Index0(pos) + offset);
	vec.push_back(Index1(pos) + offset);
	vec.push_back(Index2(pos) + offset);
	vec.push_back(Index3(pos) + offset);
}

int HorizontalSymmetric::Index0(const Position& pos) const
{
	int min = Index(pos, m_pattern & HALF);
	int max = Index(FlipHorizontal(pos), m_pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return min * m_half_size + max - (min * (min + 1) / 2);
}

DiagonalSymmetric::DiagonalSymmetric(BitBoard pattern)
	: m_pattern(pattern)
	, m_pattern_H(FlipHorizontal(pattern))
	, m_pattern_C(FlipCodiagonal(pattern))
	, m_pattern_V(FlipVertical(pattern))
	, m_half_size(Pow_int(3, PopCount(pattern & HALF)))
	, m_diag_size(Pow_int(3, PopCount(pattern & DIAG)))
{
	assert(pattern == FlipDiagonal(pattern));
}

void DiagonalSymmetric::PushBackIndices(std::vector<int>& vec, const Position& pos, std::size_t offset) const
{
	vec.push_back(Index0(pos) + offset);
	vec.push_back(Index1(pos) + offset);
	vec.push_back(Index2(pos) + offset);
	vec.push_back(Index3(pos) + offset);
}

int DiagonalSymmetric::Index0(const Position& pos) const
{
	int diag = Index(pos, m_pattern & DIAG);
	int min = Index(pos, m_pattern & HALF);
	int max = Index(FlipDiagonal(pos), m_pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return (min * m_half_size + max - (min * (min + 1) / 2)) * m_diag_size + diag;
}

Asymmetric::Asymmetric(BitBoard pattern)
	: m_pattern(pattern)
	, m_pattern_C(FlipCodiagonal(pattern))
	, m_pattern_D(FlipDiagonal(pattern))
	, m_pattern_H(FlipHorizontal(pattern))
	, m_pattern_V(FlipVertical(pattern))
	, m_patternHC(FlipCodiagonal(FlipHorizontal(pattern)))
	, m_patternHD(FlipDiagonal(FlipHorizontal(pattern)))
	, m_patternHV(FlipVertical(FlipHorizontal(pattern)))
{}

void Asymmetric::PushBackIndices(std::vector<int>& vec, const Position& pos, std::size_t offset) const
{
	vec.push_back(Index0(pos) + offset);
	vec.push_back(Index1(pos) + offset);
	vec.push_back(Index2(pos) + offset);
	vec.push_back(Index3(pos) + offset);
	vec.push_back(Index4(pos) + offset);
	vec.push_back(Index5(pos) + offset);
	vec.push_back(Index6(pos) + offset);
	vec.push_back(Index7(pos) + offset);
}

Composite::Composite(const std::vector<BitBoard>& patterns)
{
	for (const auto& p : patterns)
		index_mappers.emplace_back(CreateIndexMapper(p));
	group_order = GroupOrder();
}

std::vector<BitBoard> Composite::Patterns() const
{
	std::vector<BitBoard> ret;
	for (const auto& im : index_mappers)
	{
		auto novum = im->Patterns();
		std::move(novum.begin(), novum.end(), std::back_inserter(ret));
	}
	return ret;
}

std::vector<int> Composite::Indices(const Position& pos) const
{
	std::vector<int> ret;
	ret.reserve(group_order);
	std::size_t offset = 0;
	for (const auto& im : index_mappers)
	{
		for (const auto n : im->Indices(pos))
			ret.push_back(n + offset);
		offset += im->ReducedSize();
	}
	return ret;
}

void Composite::PushBackIndices(std::vector<int>& vec, const Position& pos, std::size_t offset) const
{
	vec.reserve(group_order);
	for (const auto& im : index_mappers)
	{
		im->PushBackIndices(vec, pos, offset);
		offset += im->ReducedSize();
	}
}

std::size_t Composite::ReducedSize() const
{
	return std::transform_reduce(index_mappers.begin(), index_mappers.end(),
								 static_cast<std::size_t>(0), std::plus<>(),
								 [](const auto& im){ return im->ReducedSize(); }
	);
}
