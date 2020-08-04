#include "ConfigIndexer.h"
#include <stdexcept>

std::unique_ptr<ConfigIndexer> CreateConfigIndexer(const BitBoard pattern)
{
	if (pattern == FlipHorizontal(pattern))
		return std::make_unique<HorizontalSymmetric>(pattern);
	if (pattern == FlipDiagonal(pattern))
		return std::make_unique<DiagonalSymmetric>(pattern);
	return std::make_unique<Asymmetric>(pattern);
}

std::unique_ptr<ConfigIndexer> CreateConfigIndexer(const std::vector<BitBoard>& patterns)
{
	return std::make_unique<Composite>(patterns);
}

class BackInsertWrapper final : public OutputIterator
{
	std::back_insert_iterator<std::vector<int>> it;
public:
	BackInsertWrapper(std::back_insert_iterator<std::vector<int>> it) : it(it) {}
	BackInsertWrapper& operator*() override { return *this; }
	BackInsertWrapper& operator++() override { ++it; return *this; }
	BackInsertWrapper& operator=(int index) override { *it = index; return *this; }
};

void ConfigIndexer::generate(std::back_insert_iterator<std::vector<int>> it, const Position& pos) const
{
	BackInsertWrapper wrapper(it);
	generate(wrapper, pos);
}

HorizontalSymmetric::HorizontalSymmetric(BitBoard pattern)
	: ConfigIndexer(4)
	, pattern(pattern)
	, half_size(Pow_int(3, popcount(pattern & HALF)))
{
	reduced_size = half_size * (half_size + 1) / 2;
	if (pattern != FlipHorizontal(pattern))
		throw std::runtime_error("Pattern has no horizontal symmetry.");
}

std::vector<BitBoard> HorizontalSymmetric::Patterns() const
{
	return {
		pattern,
		FlipCodiagonal(pattern),
		FlipDiagonal(pattern),
		FlipVertical(pattern)
	};
}

void HorizontalSymmetric::generate(OutputIterator& it, const Position& pos) const
{
	*it = Index(pos); ++it;
	*it = Index(FlipCodiagonal(pos)); ++it;
	*it = Index(FlipDiagonal(pos)); ++it;
	*it = Index(FlipVertical(pos)); ++it;
}

int HorizontalSymmetric::Index(const Position& pos) const noexcept
{
	int min = ::Index(pos, pattern & HALF);
	int max = ::Index(FlipHorizontal(pos), pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return min * half_size + max - (min * (min + 1) / 2);
}

DiagonalSymmetric::DiagonalSymmetric(BitBoard pattern)
	: ConfigIndexer(4)
	, pattern(pattern)
	, half_size(Pow_int(3, popcount(pattern & HALF)))
	, diag_size(Pow_int(3, popcount(pattern & DIAG)))
{
	reduced_size = diag_size * half_size * (half_size + 1) / 2;
	if (pattern != FlipDiagonal(pattern))
		throw std::runtime_error("Pattern has no diagonal symmetry.");
}

std::vector<BitBoard> DiagonalSymmetric::Patterns() const
{
	return {
		pattern,
		FlipCodiagonal(pattern),
		FlipHorizontal(pattern),
		FlipVertical(pattern)
	};
}

void DiagonalSymmetric::generate(OutputIterator& it, const Position& pos) const
{
	*it = Index(pos); ++it;
	*it = Index(FlipCodiagonal(pos)); ++it;
	*it = Index(FlipHorizontal(pos)); ++it;
	*it = Index(FlipVertical(pos)); ++it;
}

int DiagonalSymmetric::Index(const Position& pos) const noexcept
{
	int diag = ::Index(pos, pattern & DIAG);
	int min = ::Index(pos, pattern & HALF);
	int max = ::Index(FlipDiagonal(pos), pattern & HALF);
	if (min > max)
		std::swap(min, max);

	return (min * half_size + max - (min * (min + 1) / 2)) * diag_size + diag;
}
Asymmetric::Asymmetric(BitBoard pattern)
	: ConfigIndexer(8), pattern(pattern)
{
	reduced_size = Pow_int(3, popcount(pattern));
}

std::vector<BitBoard> Asymmetric::Patterns() const
{
	auto horizontal = FlipHorizontal(pattern);
	return {
		pattern,
		FlipCodiagonal(pattern),
		FlipDiagonal(pattern),
		horizontal,
		FlipVertical(pattern),
		FlipCodiagonal(horizontal),
		FlipDiagonal(horizontal),
		FlipVertical(horizontal)
	};
}

void Asymmetric::generate(OutputIterator& it, const Position& pos) const
{
	*it = Index(pos); ++it;
	*it = Index(FlipCodiagonal(pos)); ++it;
	*it = Index(FlipDiagonal(pos)); ++it;
	*it = Index(FlipHorizontal(pos)); ++it;
	*it = Index(FlipVertical(pos)); ++it;
	*it = Index(FlipHorizontal(FlipCodiagonal(pos))); ++it;
	*it = Index(FlipHorizontal(FlipDiagonal(pos))); ++it;
	*it = Index(FlipHorizontal(FlipVertical(pos))); ++it;
}
int Asymmetric::Index(const Position& pos) const noexcept
{
	return ::Index(pos, pattern);
}

Composite::Composite(const std::vector<BitBoard>& patterns)
	: ConfigIndexer(0)
{
	for (const auto& p : patterns)
		config_indexers.emplace_back(CreateConfigIndexer(p));

	for (const auto& ci : config_indexers)
	{
		reduced_size += ci->reduced_size;
		group_order += ci->group_order;
	}
}

std::vector<BitBoard> Composite::Patterns() const
{
	std::vector<BitBoard> ret;
	for (const auto& ci : config_indexers)
	{
		auto novum = ci->Patterns();
		std::move(novum.begin(), novum.end(), std::back_inserter(ret));
	}
	return ret;
}

class OffsetWrapper final : public OutputIterator
{
	OutputIterator& it;
public:
	int offset;

	OffsetWrapper(OutputIterator& it, int offset = 0) : it(it), offset(offset) {}
	OutputIterator& operator*() override { return *this; }
	OutputIterator& operator++() override { ++it; return *this; }
	OutputIterator& operator=(int index) override { *it = index + offset; return *this; }
};

void Composite::generate(OutputIterator& it, const Position& pos) const
{
	OffsetWrapper offsetter(it);
	for (const auto& ci : config_indexers)
	{
		ci->generate(offsetter, pos);
		offsetter.offset += ci->reduced_size;
	}
}