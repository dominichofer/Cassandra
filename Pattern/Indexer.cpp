#include "Indexer.h"
#include <stdexcept>
#include <numeric>

class HorizontalSymmetric final : public Indexer
{
	static constexpr BitBoard half = BitBoard::RightHalf();
	int half_size;
public:
	HorizontalSymmetric(BitBoard pattern) : Indexer(pattern, 4), half_size(pown(3, popcount(pattern& half)))
	{
		index_space_size = half_size * (half_size + 1) / 2;
		if (pattern != FlipHorizontal(pattern))
			throw std::runtime_error("Pattern has no horizontal symmetry.");
	}

	int DenseIndex(const Position& pos) const noexcept
	{
		int min = FastIndex(pos, pattern & half);
		int max = FastIndex(FlipHorizontal(pos), pattern & half);
		if (min > max)
			std::swap(min, max);
		return min * half_size + max - (min * (min + 1) / 2);
	}
	int DenseIndex(const Position& pos, int index) const override
	{
		switch (index)
		{
			case 0: return DenseIndex(pos);
			case 1: return DenseIndex(FlipCodiagonal(pos));
			case 2: return DenseIndex(FlipDiagonal(pos));
			case 3: return DenseIndex(FlipVertical(pos));
			default: throw std::runtime_error("Index out of range.");
		}
	}
	std::vector<BitBoard> Variations() const override
	{
		return { pattern, FlipCodiagonal(pattern), FlipDiagonal(pattern), FlipVertical(pattern) };
	}
	void InsertIndices(const Position& pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + DenseIndex(pos);
		location[1] = offset + DenseIndex(FlipCodiagonal(pos));
		location[2] = offset + DenseIndex(FlipDiagonal(pos));
		location[3] = offset + DenseIndex(FlipVertical(pos));
	}
};

class DiagonalSymmetric final : public Indexer
{
	static constexpr BitBoard half = BitBoard::StrictlyLeftLower();
	static constexpr BitBoard diag = BitBoard::DiagonalLine(0);
	int half_size, diag_size;
public:
	DiagonalSymmetric(BitBoard pattern) : Indexer(pattern, 4), half_size(pown(3, popcount(pattern& half))), diag_size(pown(3, popcount(pattern& diag)))
	{
		index_space_size = diag_size * half_size * (half_size + 1) / 2;
		if (pattern != FlipDiagonal(pattern))
			throw std::runtime_error("Pattern has no diagonal symmetry.");
	}

	int DenseIndex(const Position& pos) const noexcept
	{
		int d = FastIndex(pos, pattern & diag);
		int min = FastIndex(pos, pattern & half);
		int max = FastIndex(FlipDiagonal(pos), pattern & half);
		if (min > max)
			std::swap(min, max);
		return (min * half_size + max - (min * (min + 1) / 2)) * diag_size + d;
	}
	int DenseIndex(const Position& pos, int index) const override
	{
		switch (index)
		{
			case 0: return DenseIndex(pos);
			case 1: return DenseIndex(FlipCodiagonal(pos));
			case 2: return DenseIndex(FlipHorizontal(pos));
			case 3: return DenseIndex(FlipVertical(pos));
			default: throw std::runtime_error("Index out of range.");
		}
	}
	std::vector<BitBoard> Variations() const override
	{
		return { pattern, FlipCodiagonal(pattern), FlipHorizontal(pattern), FlipVertical(pattern) };
	}
	void InsertIndices(const Position& pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + DenseIndex(pos);
		location[1] = offset + DenseIndex(FlipCodiagonal(pos));
		location[2] = offset + DenseIndex(FlipHorizontal(pos));
		location[3] = offset + DenseIndex(FlipVertical(pos));
	}
};

class Asymmetric final : public Indexer
{
public:
	Asymmetric(BitBoard pattern) : Indexer(pattern, 8, pown(3, popcount(pattern))) {}

	int DenseIndex(const Position& pos) const noexcept
	{
		return FastIndex(pos, pattern);
	}
	int DenseIndex(const Position& pos, int index) const override
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
			default: throw std::runtime_error("Index out of range.");
		}
	}
	std::vector<BitBoard> Variations() const override
	{
		return {
			pattern,
			FlipCodiagonal(pattern),
			FlipDiagonal(pattern),
			FlipHorizontal(pattern),
			FlipVertical(pattern),
			FlipCodiagonal(FlipHorizontal(pattern)),
			FlipDiagonal(FlipHorizontal(pattern)),
			FlipVertical(FlipHorizontal(pattern))
		};
	}
	void InsertIndices(const Position& pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + DenseIndex(pos);
		location[1] = offset + DenseIndex(FlipCodiagonal(pos));
		location[2] = offset + DenseIndex(FlipDiagonal(pos));
		location[3] = offset + DenseIndex(FlipHorizontal(pos));
		location[4] = offset + DenseIndex(FlipVertical(pos));
		location[5] = offset + DenseIndex(FlipHorizontal(FlipCodiagonal(pos)));
		location[6] = offset + DenseIndex(FlipHorizontal(FlipDiagonal(pos)));
		location[7] = offset + DenseIndex(FlipHorizontal(FlipVertical(pos)));
	}
};

GroupIndexer::GroupIndexer(const std::vector<BitBoard>& pattern)
{
	variations = 0;
	index_space_size = 0;
	indexers.reserve(pattern.size());
	for (const BitBoard& p : pattern)
	{
		auto i = CreateIndexer(p);
		variations += i->variations;
		index_space_size += i->index_space_size;
		indexers.push_back(std::move(i));
	}
}

std::vector<BitBoard> GroupIndexer::Variations() const
{
	std::vector<BitBoard> ret;
	for (const auto& i : indexers)
	{
		auto novum = i->Variations();
		ret.insert(ret.end(), novum.begin(), novum.end());
	}
	return ret;
}

void GroupIndexer::InsertIndices(const Position& pos, std::span<int> location) const
{
	int offset = 0;
	for (const auto& i : indexers)
	{
		i->InsertIndices(pos, location, offset);
		location = location.subspan(i->variations);
		offset += i->index_space_size;
	}
}

std::unique_ptr<Indexer> CreateIndexer(const BitBoard pattern)
{
	if (pattern == FlipHorizontal(pattern))
		return std::make_unique<HorizontalSymmetric>(pattern);
	if (pattern == FlipDiagonal(pattern))
		return std::make_unique<DiagonalSymmetric>(pattern);
	return std::make_unique<Asymmetric>(pattern);
}
