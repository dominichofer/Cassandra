#pragma once
#include "Core/Position.h"
#include <cstdint>
#include <memory>
#include <vector>

class IndexMapper
{
public:
	IndexMapper(BitBoard pattern) : Pattern(pattern) {}
	virtual ~IndexMapper() = default;

	// number of distinct patterns within.
	std::size_t Multiplicity() const;

	virtual std::vector<BitBoard> Patterns() const = 0;
	virtual std::vector<int> ReducedIndices(Board) const = 0;

	virtual std::size_t ReducedSize() const = 0;

	const BitBoard Pattern;
};

std::unique_ptr<IndexMapper> CreateIndexMapper(BitBoard pattern);


class HorizontalSymmetric final : public IndexMapper
{
	static const inline BitBoard HALF = BitBoard{ 0x0F0F0F0F0F0F0F0Fui64 };
	const BitBoard m_pattern_C, m_pattern_V, m_pattern_D;
	const int m_half_size;

public:
	HorizontalSymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override;
	std::vector<int> ReducedIndices(Board) const override;

	std::size_t ReducedSize() const override;
private:
	int ReducedIndex0(Board) const;
	int ReducedIndex1(Board) const;
	int ReducedIndex2(Board) const;
	int ReducedIndex3(Board) const;
};

class DiagonalSymmetric final : public IndexMapper
{
	static const inline BitBoard HALF = BitBoard{ 0x0080C0E0F0F8FCFEui64 };
	static const inline BitBoard DIAG = BitBoard{ 0x8040201008040201ui64 };
	const BitBoard m_pattern_H, m_pattern_C, m_pattern_V;
	const int m_half_size, m_diag_size;

public:
	DiagonalSymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override;
	std::vector<int> ReducedIndices(Board) const override;

	std::size_t ReducedSize() const override;
private:
	int ReducedIndex0(Board) const;
	int ReducedIndex1(Board) const;
	int ReducedIndex2(Board) const;
	int ReducedIndex3(Board) const;
};

class Asymmetric final : public IndexMapper
{
	const BitBoard m_pattern_H, m_pattern_V, m_pattern_D, m_pattern_C, m_patternHV, m_patternHD, m_patternHC;

public:
	Asymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override;
	std::vector<int> ReducedIndices(Board) const override;

	std::size_t ReducedSize() const override;
private:
	int ReducedIndex0(Board) const;
	int ReducedIndex1(Board) const;
	int ReducedIndex2(Board) const;
	int ReducedIndex3(Board) const;
	int ReducedIndex4(Board) const;
	int ReducedIndex5(Board) const;
	int ReducedIndex6(Board) const;
	int ReducedIndex7(Board) const;
};
