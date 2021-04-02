#pragma once
#include "Core/Core.h"
#include "Helpers.h"
#include <cstdint>
#include <memory>
#include <vector>

// Dense index generator
// Interface
class DenseIndexer
{
public:
	int reduced_size = 0;
	int variations;

	DenseIndexer(int variations) : variations(variations) {}

	virtual BitBoard PatternVariation(int index) const = 0;
	virtual int DenseIndex(const Position&, int index) const = 0;
};

std::unique_ptr<DenseIndexer> CreateDenseIndexer(BitBoard pattern);
std::unique_ptr<DenseIndexer> CreateDenseIndexer(const std::vector<BitBoard>& patterns);

class HorizontalSymmetric final : public DenseIndexer
{
	static constexpr BitBoard half = BitBoard::RightHalf();
	const BitBoard pattern;
	const int half_size;

	int DenseIndex(const Position&) const noexcept;
public:
	HorizontalSymmetric(BitBoard pattern);

	BitBoard PatternVariation(int index) const override;
	int DenseIndex(const Position&, int index) const override;
};

class DiagonalSymmetric final : public DenseIndexer
{
	static constexpr BitBoard half = BitBoard::StrictlyLeftLower();
	static constexpr BitBoard diag = BitBoard::DiagonalLine(0);
	const BitBoard pattern;
	const int half_size, diag_size;

	int DenseIndex(const Position&) const noexcept;
public:
	DiagonalSymmetric(BitBoard pattern);

	BitBoard PatternVariation(int index) const override;
	int DenseIndex(const Position&, int index) const override;
};

class Asymmetric final : public DenseIndexer
{
	const BitBoard pattern;

	int DenseIndex(const Position&) const noexcept;
public:
	Asymmetric(BitBoard pattern);

	BitBoard PatternVariation(int index) const override;
	int DenseIndex(const Position&, int index) const override;
};

class Group final : public DenseIndexer // Composite
{
	std::vector<std::unique_ptr<DenseIndexer>> indexers;
public:
	Group(const std::vector<BitBoard>& patterns);

	BitBoard PatternVariation(int index) const override;
	int DenseIndex(const Position&, int index) const override;
};