#include "Indexer.h"
#include <stdexcept>
#include <numeric>

Indexer::Indexer(uint64_t pattern, int symmetry_group_order) noexcept
	: pattern(pattern)
	, symmetry_group_order(symmetry_group_order)
{}

std::vector<int> Indexer::Indices(Position pos) const
{
	std::vector<int> ret(symmetry_group_order, 0);
	InsertIndices(pos, ret);
	return ret;
}

// Indexer of asymmetric patterns
class A_Indexer final : public Indexer
{
public:
	A_Indexer(uint64_t pattern) : Indexer(pattern, 8)
	{
		index_space_size = pown(3, std::popcount(pattern));
	}

	int Index(Position pos) const override
	{
		return FastIndex(pos, pattern);
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlippedCodiagonal(pos));
		location[2] = offset + Index(FlippedDiagonal(pos));
		location[3] = offset + Index(FlippedHorizontal(pos));
		location[4] = offset + Index(FlippedVertical(pos));
		location[5] = offset + Index(FlippedHorizontal(FlippedCodiagonal(pos)));
		location[6] = offset + Index(FlippedHorizontal(FlippedDiagonal(pos)));
		location[7] = offset + Index(FlippedHorizontal(FlippedVertical(pos)));
	}

	std::vector<uint64_t> Variations() const override
	{
		return {
			pattern,
			FlippedCodiagonal(pattern),
			FlippedDiagonal(pattern),
			FlippedHorizontal(pattern),
			FlippedVertical(pattern),
			FlippedCodiagonal(FlippedHorizontal(pattern)),
			FlippedDiagonal(FlippedHorizontal(pattern)),
			FlippedVertical(FlippedHorizontal(pattern))
		};
	}
};

// Indexer of vertically symmetric patterns
class V_Indexer final : public Indexer
{
	static constexpr uint64_t half = 0x00000000FFFFFFFFULL; // lower half
	int half_size;
public:
	V_Indexer(uint64_t pattern) : Indexer(pattern, 4)
	{
		if (not IsVerticallySymmetric(pattern))
			throw std::runtime_error("Pattern has no vertical symmetry.");

		half_size = pown(3, std::popcount(pattern & half));
		index_space_size = half_size * (half_size + 1) / 2;
	}

	int Index(Position pos) const override
	{
		int min = FastIndex(pos, pattern & half);
		int max = FastIndex(FlippedVertical(pos), pattern & half);
		if (min > max)
			std::swap(min, max);
		return min * half_size + max - (min * (min + 1) / 2);
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlippedCodiagonal(pos));
		location[2] = offset + Index(FlippedDiagonal(pos));
		location[3] = offset + Index(FlippedHorizontal(pos));
	}

	std::vector<uint64_t> Variations() const override
	{
		return {
			pattern,
			FlippedCodiagonal(pattern),
			FlippedDiagonal(pattern),
			FlippedHorizontal(pattern)
		};
	}
};

// Indexer of diagonally symmetric patterns
class D_Indexer final : public Indexer
{
	static constexpr uint64_t half = 0x0080C0E0F0F8FCFEULL; // strictly left lower triangle
	static constexpr uint64_t diag = 0x8040201008040201ULL; // diagonal line
	int half_size, diag_size;
public:
	D_Indexer(uint64_t pattern) : Indexer(pattern, 4)
	{
		if (not IsDiagonallySymmetric(pattern))
			throw std::runtime_error("Pattern has no diagonal symmetry.");

		half_size = pown(3, std::popcount(pattern & half));
		diag_size = pown(3, std::popcount(pattern & diag));
		index_space_size = diag_size * half_size * (half_size + 1) / 2;
	}

	int Index(Position pos) const override
	{
		int d = FastIndex(pos, pattern & diag);
		int min = FastIndex(pos, pattern & half);
		int max = FastIndex(FlippedDiagonal(pos), pattern & half);
		if (min > max)
			std::swap(min, max);
		return (min * half_size + max - (min * (min + 1) / 2)) * diag_size + d;
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlippedCodiagonal(pos));
		location[2] = offset + Index(FlippedHorizontal(pos));
		location[3] = offset + Index(FlippedVertical(pos));
	}

	std::vector<uint64_t> Variations() const override
	{
		return {
			pattern,
			FlippedCodiagonal(pattern),
			FlippedHorizontal(pattern),
			FlippedVertical(pattern)
		};
	}
};

// Indexer of vertically and horizontally symmetric patterns
class VH_Indexer final : public Indexer
{
	std::vector<int> dense_index;
public:
	VH_Indexer(uint64_t pattern) : Indexer(pattern, 2)
	{
		if (not IsVerticallySymmetric(pattern))
			throw std::runtime_error("Pattern has no vertical symmetry.");
		if (not IsHorizontallySymmetric(pattern))
			throw std::runtime_error("Pattern has no horizontal symmetry.");

		dense_index = std::vector<int>(pown(3, std::popcount(pattern)), -1);
		int index = 0;
		for (Position config : Configurations(pattern))
		{
			if (dense_index[FastIndex(config, pattern)] != -1)
				continue; // already indexed

			Position var1 = config;
			Position var2 = FlippedVertical(config);
			Position var3 = FlippedHorizontal(config);
			Position var4 = FlippedHorizontal(FlippedVertical(config));

			dense_index[FastIndex(var1, pattern)] = index;
			dense_index[FastIndex(var2, pattern)] = index;
			dense_index[FastIndex(var3, pattern)] = index;
			dense_index[FastIndex(var4, pattern)] = index;
			index++;
		}

		index_space_size = index;
	}

	int Index(Position pos) const override
	{
		return dense_index[FastIndex(pos, pattern)];
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlippedDiagonal(pos));
	}

	std::vector<uint64_t> Variations() const override
	{
		return {
			pattern,
			FlippedDiagonal(pattern)
		};
	}
};

// Indexer of diagonally and codiagonally symmetric patterns
class DC_Indexer final : public Indexer
{
	std::vector<int> dense_index;
public:
	DC_Indexer(uint64_t pattern) : Indexer(pattern, 2)
	{
		if (not IsDiagonallySymmetric(pattern))
			throw std::runtime_error("Pattern has no diagonal symmetry.");
		if (not IsCodiagonallySymmetric(pattern))
			throw std::runtime_error("Pattern has no codiagonal symmetry.");

		dense_index = std::vector<int>(pown(3, std::popcount(pattern)), -1);
		int index = 0;
		for (Position config : Configurations(pattern))
		{
			if (dense_index[FastIndex(config, pattern)] != -1)
				continue; // already indexed

			Position var1 = config;
			Position var2 = FlippedDiagonal(config);
			Position var3 = FlippedCodiagonal(config);
			Position var4 = FlippedCodiagonal(FlippedDiagonal(config));

			dense_index[FastIndex(var1, pattern)] = index;
			dense_index[FastIndex(var2, pattern)] = index;
			dense_index[FastIndex(var3, pattern)] = index;
			dense_index[FastIndex(var4, pattern)] = index;
			index++;
		}

		index_space_size = index;
	}

	int Index(Position pos) const override
	{
		return dense_index[FastIndex(pos, pattern)];
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlippedVertical(pos));
	}

	std::vector<uint64_t> Variations() const override
	{
		return {
			pattern,
			FlippedVertical(pattern)
		};
	}
};

// Indexer of vertically, horizontally, diagonally and codiagonally symmetric patterns
class VHDC_Indexer final : public Indexer
{
	std::vector<int> dense_index;
public:
	VHDC_Indexer(uint64_t pattern) : Indexer(pattern, 1)
	{
		if (not IsVerticallySymmetric(pattern))
			throw std::runtime_error("Pattern has no vertical symmetry.");
		if (not IsHorizontallySymmetric(pattern))
			throw std::runtime_error("Pattern has no horizontal symmetry.");
		if (not IsDiagonallySymmetric(pattern))
			throw std::runtime_error("Pattern has no diagonal symmetry.");
		if (not IsCodiagonallySymmetric(pattern))
			throw std::runtime_error("Pattern has no codiagonal symmetry.");

		dense_index = std::vector<int>(pown(3, std::popcount(pattern)), -1);
		int index = 0;
		for (Position config : Configurations(pattern))
		{
			if (dense_index[FastIndex(config, pattern)] != -1)
				continue; // already indexed

			Position var1 = config;
			Position var2 = FlippedCodiagonal(config);
			Position var3 = FlippedDiagonal(config);
			Position var4 = FlippedHorizontal(config);
			Position var5 = FlippedVertical(config);
			Position var6 = FlippedCodiagonal(FlippedHorizontal(config));
			Position var7 = FlippedDiagonal(FlippedHorizontal(config));
			Position var8 = FlippedVertical(FlippedHorizontal(config));

			dense_index[FastIndex(var1, pattern)] = index;
			dense_index[FastIndex(var2, pattern)] = index;
			dense_index[FastIndex(var3, pattern)] = index;
			dense_index[FastIndex(var4, pattern)] = index;
			dense_index[FastIndex(var5, pattern)] = index;
			dense_index[FastIndex(var6, pattern)] = index;
			dense_index[FastIndex(var7, pattern)] = index;
			dense_index[FastIndex(var8, pattern)] = index;
			index++;
		}

		index_space_size = index;
	}

	int Index(Position pos) const override
	{
		return dense_index[FastIndex(pos, pattern)];
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
	}

	std::vector<uint64_t> Variations() const override
	{
		return {
			pattern
		};
	}
};

std::unique_ptr<Indexer> CreateIndexer(uint64_t pattern)
{
	// The symmetry group of a Reversi board is D_4.
	// A pattern can have:
	// - (v) a vertical symmetry
	// - (h) a horizontal symmetry
	// - (d) a diagonal symmetry
	// - (c) a codiagonal symmetry
	// - (90) a 90° rotational symmetry
	// - (180) a 180° rotational symmetry, which is equivalent to a point-wise symmetry
	// 
	// Each pattern falls into exactly one of these categories:
	// - no symmetry
	// - (v) only vertical symmetry
	// - (h) only horizontal symmetry
	// - (d) only diagonal symmetry
	// - (c) only codiagonal symmetry
	// - (v+h) symmetries
	// - (d+c) symmetries
	// - (v+h+d+c) symmetries
	// 
	// - (v+d) symmetries implies (v+h+d+c)
	// - (v+c) symmetries implies (v+h+d+c)
	// - (h+d) symmetries implies (v+h+d+c)
	// - (h+c) symmetries implies (v+h+d+c)

	bool v = IsVerticallySymmetric(pattern);
	bool h = IsHorizontallySymmetric(pattern);
	bool d = IsDiagonallySymmetric(pattern);
	bool c = IsCodiagonallySymmetric(pattern);

	if (v and h and d and c)
		return std::make_unique<VHDC_Indexer>(pattern);
	if (v and h)
		return std::make_unique<VH_Indexer>(pattern);
	if (d and c)
		return std::make_unique<DC_Indexer>(pattern);
	if (v)
		return std::make_unique<V_Indexer>(pattern);
	if (h)
		return std::make_unique<V_Indexer>(FlippedDiagonal(pattern));
	if (d)
		return std::make_unique<D_Indexer>(pattern);
	if (c)
		return std::make_unique<D_Indexer>(FlippedVertical(pattern));
	return std::make_unique<A_Indexer>(pattern);
}

std::size_t ConfigurationsOfPattern(uint64_t pattern)
{
	return CreateIndexer(pattern)->index_space_size;
}

std::size_t ConfigurationsOfPattern(std::vector<uint64_t> pattern)
{
	std::size_t size = 0;
	for (uint64_t p : pattern)
		size += ConfigurationsOfPattern(p);
	return size;
}



GroupIndexer::GroupIndexer(std::vector<uint64_t> pattern)
{
	index_space_size = 0;
	indexers.reserve(pattern.size());
	for (uint64_t p : pattern)
	{
		auto i = CreateIndexer(p);
		index_space_size += i->index_space_size;
		indexers.push_back(std::move(i));
	}
}

std::vector<uint64_t> GroupIndexer::Variations() const
{
	std::vector<uint64_t> ret;
	for (const auto& i : indexers)
	{
		auto novum = i->Variations();
		ret.insert(ret.end(), novum.begin(), novum.end());
	}
	return ret;
}


std::vector<int> GroupIndexer::Indices(Position pos) const
{
	std::vector<int> ret(Variations().size(), 0);
	InsertIndices(pos, ret);
	return ret;
}

void GroupIndexer::InsertIndices(Position pos, std::span<int> location) const
{
	int offset = 0;
	for (const auto& i : indexers)
	{
		i->InsertIndices(pos, location, offset);
		location = location.subspan(i->symmetry_group_order);
		offset += i->index_space_size;
	}
}
