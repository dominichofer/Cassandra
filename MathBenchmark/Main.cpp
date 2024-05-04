#include "benchmark/benchmark.h"
#include "Math/Math.h"
#include "IO/IO.h"
#include <cstdint>
#include <random>

class BenchmarkTable : public Table
{
public:
	BenchmarkTable() : Table(
		"        name        |  size   |  time  |  MB   |GFlop/s|GByte/s",
		"{:20}|{:9L}|{:5}|{:7L}|{:7L}|{:7L}"
	) {}

	void PrintRow(const std::string& name, int64_t size, auto duration, int64_t flops, int64_t bytes) const
	{
		static std::string last_name = "";
		Table::PrintRow(
			name == last_name ? "" : name,
			size,
			ShortTimeString(duration),
			bytes / 1024 / 1024,
			flops / duration.count(),
			bytes / duration.count()
		);
		last_name = name;
	}
};

Matrix RandomMatrix(std::size_t rows, std::size_t cols)
{
	std::mt19937_64 rng;
	std::uniform_real_distribution<double> dist{ -1, 1 };

	Matrix mat(cols, rows);
	for (std::size_t i = 0; i < rows; i++)
		for (std::size_t j = 0; j < cols; j++)
			mat(i, j) = dist(rng);
	return mat;
}

MatrixCSR RandomMatrixCSR(std::size_t elements_per_row, std::size_t rows, std::size_t cols)
{
	std::mt19937_64 rng;
	std::uniform_int_distribution<uint32_t> dist{ 0, static_cast<uint32_t>(cols) - 1 };

	MatrixCSR mat(elements_per_row, rows, cols);
	for (std::size_t i = 0; i < mat.Rows(); i++)
		for (auto& elem : mat.Row(i))
			elem = dist(rng);
	return mat;
}

Vector RandomVector(std::size_t size)
{
	std::mt19937_64 rng;
	std::uniform_real_distribution<double> dist{ -1, 1 };

	Vector vec(size, 0.0);
	for (std::size_t i = 0; i < size; i++)
		vec[i] = dist(rng);
	return vec;
}

void DenseMatVec(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	for (int size = 1; size <= 8 * 1024; size *= 2)
	{
		auto A = RandomMatrix(size, size);
		auto x = RandomVector(size);
		benchmark::DoNotOptimize(A * x); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(A * x);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * size * size, sizeof(double) * (size * size + size));
	}
	table.PrintSeparator();
}

void DenseVecMat(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	for (int size = 1; size <= 8 * 1024; size *= 2)
	{
		auto A = RandomMatrix(size, size);
		auto x = RandomVector(size);
		benchmark::DoNotOptimize(x * A); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(x * A);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * size * size, sizeof(double) * (size * size + size));
	}
	table.PrintSeparator();
}

void CSRMatVec(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	auto elements_per_row = 32;
	for (int size = 1024; size <= 2'097'152; size *= 2)
	{
		auto A = RandomMatrixCSR(elements_per_row, size, size / 16);
		auto x = RandomVector(size / 16);
		benchmark::DoNotOptimize(A * x); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(A * x);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * elements_per_row * size, sizeof(uint32_t) * elements_per_row * size + sizeof(double) * size / 16);
	}
	table.PrintSeparator();
}

void CSRVecMat(const BenchmarkTable& table, const std::string& name)
{
	table.PrintHeader();
	auto elements_per_row = 32;
	for (int size = 1024; size <= 2'097'152; size *= 2)
	{
		auto A = RandomMatrixCSR(elements_per_row, size, size / 16);
		auto x = RandomVector(size);
		benchmark::DoNotOptimize(x * A); // warm up
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			benchmark::DoNotOptimize(x * A);
		auto stop = std::chrono::high_resolution_clock::now();
		table.PrintRow(name, size, (stop - start) / 100, 2 * elements_per_row * size, sizeof(uint32_t) * elements_per_row * size + sizeof(double) * size);
	}
	table.PrintSeparator();
}

int main()
{
	BenchmarkTable table;
	DenseMatVec(table, "dense A*x");
	DenseVecMat(table, "dense x*A");
	CSRMatVec(table, "CSR A*x");
	CSRVecMat(table, "CSR x*A");
}