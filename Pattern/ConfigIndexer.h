#pragma once
#include "Core/Core.h"
#include "Helpers.h"
#include <cstdint>
#include <memory>
#include <vector>

class OutputIterator
{
public:
	virtual OutputIterator& operator*() = 0;
	virtual OutputIterator& operator++() = 0;
	virtual OutputIterator& operator=(int) = 0;
};

// Reduced configuration index generator
class ConfigIndexer
{
public:
	int reduced_size = 0;
	int group_order; // number of unique patterns within.

	ConfigIndexer(int group_order) : group_order(group_order) {}

	virtual std::vector<BitBoard> Patterns() const = 0;

	virtual void generate(OutputIterator&, const Position&) const = 0;
	void generate(std::back_insert_iterator<std::vector<int>>, const Position&) const;
};

std::unique_ptr<ConfigIndexer> CreateConfigIndexer(BitBoard pattern);
std::unique_ptr<ConfigIndexer> CreateConfigIndexer(const std::vector<BitBoard>& patterns);


// TODO: Refactor into Vertical symmetrie for performance reasons?
class HorizontalSymmetric final : public ConfigIndexer
{
	static constexpr BitBoard half = BitBoard::LowerHalf();
	const BitBoard pattern;
	const int half_size;

	int Index(const Position&) const noexcept;
public:
	HorizontalSymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override;
	void generate(OutputIterator&, const Position&) const override;
};

class DiagonalSymmetric final : public ConfigIndexer
{
	static constexpr BitBoard half = BitBoard::StrictlyLeftLower();
	static constexpr BitBoard diag = BitBoard::DiagonalLine(0);
	const BitBoard pattern;
	const int half_size, diag_size;

	int Index(const Position&) const noexcept;
public:
	DiagonalSymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const;
	void generate(OutputIterator&, const Position&) const override;
};

class Asymmetric final : public ConfigIndexer
{
	const BitBoard pattern;

	int Index(const Position&) const noexcept;
public:
	Asymmetric(BitBoard pattern);

	std::vector<BitBoard> Patterns() const override;
	void generate(OutputIterator&, const Position&) const override;
};

class Composite final : public ConfigIndexer
{
	std::vector<std::unique_ptr<ConfigIndexer>> config_indexers;
public:
	Composite(const std::vector<BitBoard>& patterns);

	std::vector<BitBoard> Patterns() const override;
	void generate(OutputIterator&, const Position&) const override;
};