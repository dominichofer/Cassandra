#include "Indexer.h"
#include <stdexcept>
#include <numeric>

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
	A_Indexer(BitBoard pattern) : Indexer(pattern, 8)
	{
		index_space_size = pown(3, popcount(pattern));
	}

	int Index(Position pos) const override
	{
		return FastIndex(pos, pattern);
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlipCodiagonal(pos));
		location[2] = offset + Index(FlipDiagonal(pos));
		location[3] = offset + Index(FlipHorizontal(pos));
		location[4] = offset + Index(FlipVertical(pos));
		location[5] = offset + Index(FlipHorizontal(FlipCodiagonal(pos)));
		location[6] = offset + Index(FlipHorizontal(FlipDiagonal(pos)));
		location[7] = offset + Index(FlipHorizontal(FlipVertical(pos)));
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
};

// Indexer of vertically symmetric patterns
class V_Indexer final : public Indexer
{
	static constexpr BitBoard half = BitBoard::LowerHalf();
	int half_size;
public:
	V_Indexer(BitBoard pattern) : Indexer(pattern, 4)
	{
		if (not pattern.IsVerticallySymmetric())
			throw std::runtime_error("Pattern has no vertical symmetry.");

		half_size = pown(3, popcount(pattern & half));
		index_space_size = half_size * (half_size + 1) / 2;
	}

	int Index(Position pos) const override
	{
		int min = FastIndex(pos, pattern & half);
		int max = FastIndex(FlipVertical(pos), pattern & half);
		if (min > max)
			std::swap(min, max);
		return min * half_size + max - (min * (min + 1) / 2);
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlipCodiagonal(pos));
		location[2] = offset + Index(FlipDiagonal(pos));
		location[3] = offset + Index(FlipHorizontal(pos));
	}

	std::vector<BitBoard> Variations() const override
	{
		return {
			pattern,
			FlipCodiagonal(pattern),
			FlipDiagonal(pattern),
			FlipHorizontal(pattern)
		};
	}
};

// Indexer of diagonally symmetric patterns
class D_Indexer final : public Indexer
{
	static constexpr BitBoard half = BitBoard::StrictlyLeftLowerTriangle();
	static constexpr BitBoard diag = BitBoard::DiagonalLine(0);
	int half_size, diag_size;
public:
	D_Indexer(BitBoard pattern) : Indexer(pattern, 4)
	{
		if (not pattern.IsDiagonallySymmetric())
			throw std::runtime_error("Pattern has no diagonal symmetry.");

		half_size = pown(3, popcount(pattern & half));
		diag_size = pown(3, popcount(pattern & diag));
		index_space_size = diag_size * half_size * (half_size + 1) / 2;
	}

	int Index(Position pos) const override
	{
		int d = FastIndex(pos, pattern & diag);
		int min = FastIndex(pos, pattern & half);
		int max = FastIndex(FlipDiagonal(pos), pattern & half);
		if (min > max)
			std::swap(min, max);
		return (min * half_size + max - (min * (min + 1) / 2)) * diag_size + d;
	}

	void InsertIndices(Position pos, std::span<int> location, int offset) const override
	{
		location[0] = offset + Index(pos);
		location[1] = offset + Index(FlipCodiagonal(pos));
		location[2] = offset + Index(FlipHorizontal(pos));
		location[3] = offset + Index(FlipVertical(pos));
	}

	std::vector<BitBoard> Variations() const override
	{
		return {
			pattern,
			FlipCodiagonal(pattern),
			FlipHorizontal(pattern),
			FlipVertical(pattern)
		};
	}
};

// Indexer of vertically and horizontally symmetric patterns
class VH_Indexer final : public Indexer
{
	std::vector<int> dense_index;
public:
	VH_Indexer(BitBoard pattern) : Indexer(pattern, 2)
	{
		if (not pattern.IsVerticallySymmetric())
			throw std::runtime_error("Pattern has no vertical symmetry.");
		if (not pattern.IsHorizontallySymmetric())
			throw std::runtime_error("Pattern has no horizontal symmetry.");

		dense_index = std::vector<int>(pown(3, popcount(pattern)), -1);
		int index = 0;
		for (Position config : Configurations(pattern))
		{
			if (dense_index[FastIndex(config, pattern)] != -1)
				continue; // already indexed

			Position var1 = config;
			Position var2 = FlipVertical(config);
			Position var3 = FlipHorizontal(config);
			Position var4 = FlipHorizontal(FlipVertical(config));

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
		location[1] = offset + Index(FlipDiagonal(pos));
	}

	std::vector<BitBoard> Variations() const override
	{
		return {
			pattern,
			FlipDiagonal(pattern)
		};
	}
};

// Indexer of diagonally and codiagonally symmetric patterns
class DC_Indexer final : public Indexer
{
	std::vector<int> dense_index;
public:
	DC_Indexer(BitBoard pattern) : Indexer(pattern, 2)
	{
		if (not pattern.IsDiagonallySymmetric())
			throw std::runtime_error("Pattern has no diagonal symmetry.");
		if (not pattern.IsCodiagonallySymmetric())
			throw std::runtime_error("Pattern has no codiagonal symmetry.");

		dense_index = std::vector<int>(pown(3, popcount(pattern)), -1);
		int index = 0;
		for (Position config : Configurations(pattern))
		{
			if (dense_index[FastIndex(config, pattern)] != -1)
				continue; // already indexed

			Position var1 = config;
			Position var2 = FlipDiagonal(config);
			Position var3 = FlipCodiagonal(config);
			Position var4 = FlipCodiagonal(FlipDiagonal(config));

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
		location[1] = offset + Index(FlipVertical(pos));
	}

	std::vector<BitBoard> Variations() const override
	{
		return {
			pattern,
			FlipVertical(pattern)
		};
	}
};

// Indexer of vertically, horizontally, diagonally and codiagonally symmetric patterns
class VHDC_Indexer final : public Indexer
{
	std::vector<int> dense_index;
public:
	VHDC_Indexer(BitBoard pattern) : Indexer(pattern, 1)
	{
		if (not pattern.IsVerticallySymmetric())
			throw std::runtime_error("Pattern has no vertical symmetry.");
		if (not pattern.IsHorizontallySymmetric())
			throw std::runtime_error("Pattern has no horizontal symmetry.");
		if (not pattern.IsDiagonallySymmetric())
			throw std::runtime_error("Pattern has no diagonal symmetry.");
		if (not pattern.IsCodiagonallySymmetric())
			throw std::runtime_error("Pattern has no codiagonal symmetry.");

		dense_index = std::vector<int>(pown(3, popcount(pattern)), -1);
		int index = 0;
		for (Position config : Configurations(pattern))
		{
			if (dense_index[FastIndex(config, pattern)] != -1)
				continue; // already indexed

			Position var1 = config;
			Position var2 = FlipCodiagonal(config);
			Position var3 = FlipDiagonal(config);
			Position var4 = FlipHorizontal(config);
			Position var5 = FlipVertical(config);
			Position var6 = FlipCodiagonal(FlipHorizontal(config));
			Position var7 = FlipDiagonal(FlipHorizontal(config));
			Position var8 = FlipVertical(FlipHorizontal(config));

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

	std::vector<BitBoard> Variations() const override
	{
		return {
			pattern
		};
	}
};

std::unique_ptr<Indexer> CreateIndexer(BitBoard pattern)
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

	bool v = pattern.IsVerticallySymmetric();
	bool h = pattern.IsHorizontallySymmetric();
	bool d = pattern.IsDiagonallySymmetric();
	bool c = pattern.IsCodiagonallySymmetric();

	if (v and h and d and c)
		return std::make_unique<VHDC_Indexer>(pattern);
	if (v and h)
		return std::make_unique<VH_Indexer>(pattern);
	if (d and c)
		return std::make_unique<DC_Indexer>(pattern);
	if (v)
		return std::make_unique<V_Indexer>(pattern);
	if (h)
		return std::make_unique<V_Indexer>(FlipDiagonal(pattern));
	if (d)
		return std::make_unique<D_Indexer>(pattern);
	if (c)
		return std::make_unique<D_Indexer>(FlipVertical(pattern));
	return std::make_unique<A_Indexer>(pattern);
}



GroupIndexer::GroupIndexer(std::vector<BitBoard> pattern)
{
	index_space_size = 0;
	indexers.reserve(pattern.size());
	for (BitBoard p : pattern)
	{
		auto i = CreateIndexer(p);
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
