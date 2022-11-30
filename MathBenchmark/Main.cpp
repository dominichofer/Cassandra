#include "benchmark/benchmark.h"
#include "Math/Math.h"
#include "IO/IO.h"
#include <random>

class BenchmarkTable : public Table
{
public:
	BenchmarkTable() : Table(
		"        name        |  size   |  time  |  MB   |GFlop/s|GByte/s",
		"{:20}|{:9L}|{:5}|{:7L}|{:7L}|{:7L}"
	) {}

	void PrintRow(const std::string& name, int64 size, auto duration, int64 flops, int64 bytes) const
	{
		static std::string last_name = "";
		Table::PrintRow(
			name == last_name ? std::nullopt : std::optional(name),
			size,
			short_time_format(duration),
			bytes / 1024 / 1024,
			flops / duration.count(),
			bytes / duration.count()
		);
		last_name = name;
	}
};

template <typename T>
DenseMatrix<T> RandomMatrix(std::size_t rows, std::size_t cols)
{
	std::mt19937_64 rng;
	std::uniform_real_distribution<T> dist{ -1, 1 };

	DenseMatrix<T> mat(rows, cols);
	for (std::size_t i = 0; i < rows; i++)
		for (std::size_t j = 0; j < cols; j++)
			mat(i, j) = dist(rng);
	return mat;
}

template <typename T>
MatrixCSR<T> RandomMatrixCSR(std::size_t elements_per_row, std::size_t rows, T cols)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<T> dist{ 0, cols - 1 };

	MatrixCSR<T> mat(elements_per_row, cols, rows);
	for (auto& elem : mat)
		elem = dist(rng);
	return mat;
}

template <typename T>
std::vector<T> RandomVector(std::size_t size)
{
	std::mt19937_64 rng;
	std::uniform_real_distribution<T> dist{ -1, 1 };

	std::vector<T> vec(size);
	for (std::size_t i = 0; i < size; i++)
		vec[i] = dist(rng);
	return vec;
}

template <typename T>
void DenseMatVec(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	for (int size = 1; size <= 8 * 1024; size *= 2)
	{
		auto A = RandomMatrix<T>(size, size);
		auto x = RandomVector<T>(size);
		benchmark::DoNotOptimize(A * x); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(A * x);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * size * size, sizeof(T) * (size * size + size));
	}
	table.PrintSeparator();
}

template <typename T>
void DenseVecMat(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	for (int size = 1; size <= 8 * 1024; size *= 2)
	{
		auto A = RandomMatrix<T>(size, size);
		auto x = RandomVector<T>(size);
		benchmark::DoNotOptimize(x * A); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(x * A);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * size * size, sizeof(T) * (size * size + size));
	}
	table.PrintSeparator();
}

template <typename IndexType, typename T>
void CSRMatVec(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	auto elements_per_row = 32;
	for (int size = 1024; size <= 2_MB; size *= 2)
	{
		auto A = RandomMatrixCSR<IndexType>(elements_per_row, size, size / 16);
		auto x = RandomVector<T>(size / 16);
		benchmark::DoNotOptimize(A * x); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(A * x);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * elements_per_row * size, sizeof(IndexType) * elements_per_row * size + sizeof(T) * size / 16);
	}
	table.PrintSeparator();
}

template <typename IndexType, typename T>
void CSRVecMat(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	auto elements_per_row = 32;
	for (int size = 1024; size <= 2_MB; size *= 2)
	{
		auto A = RandomMatrixCSR<IndexType>(elements_per_row, size, size / 16);
		auto x = RandomVector<T>(size);
		benchmark::DoNotOptimize(x * A); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(x * A);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * elements_per_row * size, sizeof(IndexType) * elements_per_row * size + sizeof(T) * size);
	}
	table.PrintSeparator();
}

int main()
{
	BenchmarkTable table;
	DenseMatVec<float>(table, "dense A*x float");
	DenseMatVec<double>(table, "dense A*x double");
	DenseVecMat<float>(table, "dense x*A float");
	DenseVecMat<double>(table, "dense x*A double");
	CSRMatVec<int32, float>(table, "CSR A*x int32,float");
	CSRMatVec<int32, double>(table, "CSR A*x int32,double");
	CSRVecMat<int32, float>(table, "CSR x*A int32,float");
	CSRVecMat<int32, double>(table, "CSR x*A int32,double");
}