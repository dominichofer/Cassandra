#pragma once
#include "Core/Position.h"
#include "Helpers.h"
#include "Machine/BitTwiddling.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <any>

class OutputIterator
{
public:
	virtual OutputIterator& operator*() = 0;
	virtual OutputIterator& operator++() = 0;
	virtual OutputIterator& operator=(int) = 0;
};

// Reduced configuration index generator
class IndexMapper
{
public:
	// number of unique patterns within.
	std::size_t GroupOrder() const;

	virtual std::vector<BitBoard> Patterns() const = 0;
	virtual std::vector<int> Indices(const Position& pos) const = 0;

	virtual std::size_t ReducedSize() const = 0;

	virtual void generate(OutputIterator&, const Position&) const = 0;
};

std::unique_ptr<IndexMapper> CreateIndexMapper(BitBoard pattern);
std::unique_ptr<IndexMapper> CreateIndexMapper(const std::vector<BitBoard>& patterns);


// TODO: Refactor into Vertical symmetrie for performance reasons?
class HorizontalSymmetric final : public IndexMapper
{
	static constexpr BitBoard HALF = BitBoard{ 0x0F0F0F0F0F0F0F0FULL };
	const BitBoard m_pattern, m_pattern_C, m_pattern_D, m_pattern_V;
	const int m_half_size;

public:
	HorizontalSymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override { return { m_pattern, m_pattern_C, m_pattern_D, m_pattern_V }; }
	std::vector<int> Indices(const Position& pos) const override { return { Index0(pos), Index1(pos), Index2(pos), Index3(pos) }; }
	void generate(OutputIterator&, const Position&) const override;

	std::size_t ReducedSize() const override { return m_half_size * (m_half_size + 1) / 2; }
private:
	int Index0(const Position& pos) const;
	int Index1(const Position& pos) const { return Index0(FlipCodiagonal(pos)); }
	int Index2(const Position& pos) const { return Index0(FlipDiagonal(pos));  }
	int Index3(const Position& pos) const { return Index0(FlipVertical(pos));  }
};

class DiagonalSymmetric final : public IndexMapper
{
	static constexpr BitBoard HALF = BitBoard{ 0x0080C0E0F0F8FCFEULL };
	static constexpr BitBoard DIAG = BitBoard{ 0x8040201008040201ULL };
	const BitBoard m_pattern, m_pattern_C, m_pattern_H, m_pattern_V;
	const int m_half_size, m_diag_size;

public:
	DiagonalSymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override { return { m_pattern, m_pattern_C, m_pattern_H, m_pattern_V }; }
	std::vector<int> Indices(const Position& pos) const override { return { Index0(pos), Index1(pos), Index2(pos), Index3(pos) }; }
	void generate(OutputIterator&, const Position&) const override;

	std::size_t ReducedSize() const override { return m_diag_size * m_half_size * (m_half_size + 1) / 2; }
private:
	int Index0(const Position& pos) const;
	int Index1(const Position& pos) const { return Index0(FlipCodiagonal(pos)); }
	int Index2(const Position& pos) const { return Index0(FlipHorizontal(pos)); }
	int Index3(const Position& pos) const { return Index0(FlipVertical(pos));  }
};

class Asymmetric final : public IndexMapper
{
	const BitBoard m_pattern, m_pattern_C, m_pattern_D, m_pattern_H, m_pattern_V, m_patternHC, m_patternHD, m_patternHV;

public:
	Asymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override { return { m_pattern, m_pattern_C, m_pattern_D, m_pattern_H, m_pattern_V, m_patternHC, m_patternHD, m_patternHV }; }
	std::vector<int> Indices(const Position& pos) const override { return { Index0(pos), Index1(pos), Index2(pos), Index3(pos), Index4(pos), Index5(pos), Index6(pos), Index7(pos) };}
	void generate(OutputIterator&, const Position&) const override;

	std::size_t ReducedSize() const override { return Pow_int(3, PopCount(m_pattern)); }
private:
	int Index0(const Position& pos) const { return Index(pos, m_pattern); }
	int Index1(const Position& pos) const { return Index0(FlipCodiagonal(pos)); }
	int Index2(const Position& pos) const { return Index0(FlipDiagonal(pos)); }
	int Index3(const Position& pos) const { return Index0(FlipHorizontal(pos)); }
	int Index4(const Position& pos) const { return Index0(FlipVertical(pos)); }
	int Index5(const Position& pos) const { return Index0(FlipHorizontal(FlipCodiagonal(pos))); }
	int Index6(const Position& pos) const { return Index0(FlipHorizontal(FlipDiagonal(pos))); }
	int Index7(const Position& pos) const { return Index0(FlipHorizontal(FlipVertical(pos))); }
};

class Composite final : public IndexMapper
{
	std::vector<std::unique_ptr<IndexMapper>> index_mappers;
	std::size_t group_order;
public:
	Composite(const std::vector<BitBoard>& patterns);

	std::vector<BitBoard> Patterns() const override;
	std::vector<int> Indices(const Position&) const override;
	void generate(OutputIterator&, const Position&) const override;

	std::size_t ReducedSize() const override;
};