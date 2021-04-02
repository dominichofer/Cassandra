#include "DenseIndexer.h"
#include <stdexcept>

std::unique_ptr<DenseIndexer> CreateDenseIndexer(const BitBoard pattern)
{
	if (pattern == FlipHorizontal(pattern))
		return std::make_unique<HorizontalSymmetric>(pattern);
	if (pattern == FlipDiagonal(pattern))
		return std::make_unique<DiagonalSymmetric>(pattern);
	return std::make_unique<Asymmetric>(pattern);
}

std::unique_ptr<DenseIndexer> CreateDenseIndexer(const std::vector<BitBoard>& patterns)
{
	return std::make_unique<Group>(patterns);
}


HorizontalSymmetric::HorizontalSymmetric(BitBoard pattern)
	: DenseIndexer(4)
	, pattern(pattern)
	, half_size(pown(3, popcount(pattern & half)))
{
	reduced_size = half_size * (half_size + 1) / 2;
	if (pattern != FlipHorizontal(pattern))
		throw std::runtime_error("Pattern has no horizontal symmetry.");
}

int HorizontalSymmetric::DenseIndex(const Position& pos) const noexcept
{
	int min = FastIndex(pos, pattern & half);
	int max = FastIndex(FlipHorizontal(pos), pattern & half);
	if (min > max)
		std::swap(min, max);
	return min * half_size + max - (min * (min + 1) / 2);
}

BitBoard HorizontalSymmetric::PatternVariation(int index) const
{
	switch (index)
	{
		case 0: return pattern;
		case 1: return FlipCodiagonal(pattern);
		case 2: return FlipDiagonal(pattern);
		case 3: return FlipVertical(pattern);
	}
	throw std::runtime_error("Index out of range.");
}

int HorizontalSymmetric::DenseIndex(const Position& pos, int index) const
{
	switch (index)
	{
		case 0: return DenseIndex(pos);
		case 1: return DenseIndex(FlipCodiagonal(pos));
		case 2: return DenseIndex(FlipDiagonal(pos));
		case 3: return DenseIndex(FlipVertical(pos));
	}
	throw std::runtime_error("Index out of range.");
}


DiagonalSymmetric::DiagonalSymmetric(BitBoard pattern)
	: DenseIndexer(4)
	, pattern(pattern)
	, half_size(pown(3, popcount(pattern & half)))
	, diag_size(pown(3, popcount(pattern & diag)))
{
	reduced_size = diag_size * half_size * (half_size + 1) / 2;
	if (pattern != FlipDiagonal(pattern))
		throw std::runtime_error("Pattern has no diagonal symmetry.");
}

int DiagonalSymmetric::DenseIndex(const Position& pos) const noexcept
{
	int d = FastIndex(pos, pattern & diag);
	int min = FastIndex(pos, pattern & half);
	int max = FastIndex(FlipDiagonal(pos), pattern & half);
	if (min > max)
		std::swap(min, max);
	return (min * half_size + max - (min * (min + 1) / 2)) * diag_size + d;
}

BitBoard DiagonalSymmetric::PatternVariation(int index) const
{
	switch (index)
	{
		case 0: return pattern;
		case 1: return FlipCodiagonal(pattern);
		case 2: return FlipHorizontal(pattern);
		case 3: return FlipVertical(pattern);
	}
	throw std::runtime_error("Index out of range.");
}

int DiagonalSymmetric::DenseIndex(const Position& pos, int index) const
{
	switch (index)
	{
		case 0: return DenseIndex(pos);
		case 1: return DenseIndex(FlipCodiagonal(pos));
		case 2: return DenseIndex(FlipHorizontal(pos));
		case 3: return DenseIndex(FlipVertical(pos));
	}
	throw std::runtime_error("Index out of range.");
}

Asymmetric::Asymmetric(BitBoard pattern)
	: DenseIndexer(8), pattern(pattern)
{
	reduced_size = pown(3, popcount(pattern));
}

int Asymmetric::DenseIndex(const Position& pos) const noexcept
{
	return FastIndex(pos, pattern);
}

BitBoard Asymmetric::PatternVariation(int index) const
{
	switch (index)
	{
		case 0: return pattern;
		case 1: return FlipCodiagonal(pattern);
		case 2: return FlipDiagonal(pattern);
		case 3: return FlipHorizontal(pattern);
		case 4: return FlipVertical(pattern);
		case 5: return FlipCodiagonal(FlipHorizontal(pattern));
		case 6: return FlipDiagonal(FlipHorizontal(pattern));
		case 7: return FlipVertical(FlipHorizontal(pattern));
	}
	throw std::runtime_error("Index out of range.");
}

int Asymmetric::DenseIndex(const Position& pos, int index) const
{
	switch (index)
	{
		case 0: return DenseIndex(pos);
		case 1: return DenseIndex(FlipCodiagonal(pos));
		case 2: return DenseIndex(FlipDiagonal(pos));
		case 3: return DenseIndex(FlipHorizontal(pos));
		case 4: return DenseIndex(FlipVertical(pos));
		case 5: return DenseIndex(FlipHorizontal(FlipCodiagonal(pos)));
		case 6: return DenseIndex(FlipHorizontal(FlipDiagonal(pos)));
		case 7: return DenseIndex(FlipHorizontal(FlipVertical(pos)));
	}
	throw std::runtime_error("Index out of range.");
}

Group::Group(const std::vector<BitBoard>& patterns)
	: DenseIndexer(0)
{
	for (const BitBoard& p : patterns)
	{
		auto indexer = CreateDenseIndexer(p);
		reduced_size += indexer->reduced_size;
		variations += indexer->variations;
		indexers.push_back(std::move(indexer));
	}
}

BitBoard Group::PatternVariation(int index) const
{
	for (const auto& i : indexers)
	{
		if (index < i->variations)
			return i->PatternVariation(index);
		index -= i->variations;
	}
	throw std::runtime_error("Index out of range.");
}

int Group::DenseIndex(const Position& pos, int index) const
{
	int offset = 0;
	for (const auto& i : indexers)
	{
		if (index < i->variations)
			return offset + i->DenseIndex(pos, index);
		offset += i->reduced_size;
		index -= i->variations;
	}
	throw std::runtime_error("Index out of range.");
}