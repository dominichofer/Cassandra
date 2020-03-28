#include "Core/Position.h"
#include "Core/PositionGenerator.h"
#include "Engine/PVSearch.h"
#include "Pattern/ConfigIndexer.h"
#include "Math/Matrix.h"
#include "Math/MatrixCSR.h"
#include "Math/Vector.h"
#include "Math/Solver.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>
#include <string>
#include <chrono>

constexpr BitBoard L0 = 0x00000000000000FFULL;
constexpr BitBoard L1 = 0x000000000000FF00ULL;
constexpr BitBoard L2 = 0x0000000000FF0000ULL;
constexpr BitBoard L3 = 0x00000000FF000000ULL;
constexpr BitBoard D3 = 0x0000000000010204ULL;
constexpr BitBoard D4 = 0x0000000001020408ULL;
constexpr BitBoard D5 = 0x0000000102040810ULL;
constexpr BitBoard D6 = 0x0000010204081020ULL;
constexpr BitBoard D7 = 0x0001020408102040ULL;
constexpr BitBoard D8 = 0x0102040810204080ULL;
constexpr BitBoard B5 = 0x0000000000001F1FULL;
constexpr BitBoard Q0 = 0x0000000000070707ULL;
constexpr BitBoard Q1 = 0x0000000000070707ULL << 9;
constexpr BitBoard Q2 = 0x0000000000070707ULL << 18;
constexpr BitBoard Ep = 0x0000000000003CBDULL;
constexpr BitBoard Epp = 0x0000000000003CFFULL;
constexpr BitBoard C3 = 0x0000000000010307ULL;
constexpr BitBoard C4 = 0x000000000103070FULL;
constexpr BitBoard C4p = 0x000000000107070FULL;
constexpr BitBoard C3p1 = 0x000000000101030FULL;
constexpr BitBoard C3p2 = 0x000000010101031FULL;
constexpr BitBoard C3p3 = 0x000001010101033FULL;
constexpr BitBoard C4p1 = 0x000000010103071FULL;
constexpr BitBoard Comet = 0x8040201008040303ULL;
constexpr BitBoard Cometp = 0xC0C0201008040303ULL;
constexpr BitBoard C3pp= 0x81010000000103C7ULL;
constexpr BitBoard C3ppp= 0x81410000000103C7ULL;
constexpr BitBoard C4pp = C4 | C3pp;
constexpr BitBoard AA = 0x000000010105031FULL;

template <typename Iterator>
class IteratorWrapper final : public OutputIterator
{
	Iterator it;
public:
	IteratorWrapper(Iterator it) : it(it) {}
	IteratorWrapper& operator*() override { return *this; }
	IteratorWrapper& operator++() override { ++it; return *this; }
	IteratorWrapper& operator=(int index) override { *it = index; return *this; }
};

auto CreateMatrix(const ConfigIndexer& config_indexer, const std::vector<Position>& positions)
{
	const auto entries_per_row = config_indexer.group_order;
	const auto cols = config_indexer.reduced_size;
	const auto rows = positions.size();
	MatrixCSR<uint32_t> mat(entries_per_row, cols, rows);

	const int64_t size = positions.size();
	#pragma omp parallel for schedule(dynamic, 64)
	for (int64_t i = 0; i < size; i++)
	{
		IteratorWrapper output_it(mat.begin() + i * entries_per_row);
		config_indexer.generate(output_it, positions[i]);
	}
	return mat;
}

int main(int argc, char* argv[])
{
	//auto config_indexer = CreateConfigIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, Ep, C3p1, B5 }); // 6.38253
	//auto config_indexer = CreateConfigIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, Epp, C3p1, B5 }); // 6.46631
	//auto config_indexer = CreateConfigIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, C3p1, B5 }); //6.38712
	//auto config_indexer = CreateConfigIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, C3p1, B6 }); // 6.5572

	HashTablePVS tt{ 1'000'000 };
	Search::PVSearch pvs{ tt };

	std::vector<Position> train_positions;
	std::vector<Position> test_positions;
	for (int e = 5; e < 10; e++)
		std::generate_n(std::back_inserter(train_positions), 1'000'000, PosGen::Random_with_empty_count(e /*empty_count*/, 13));
	for (int e = 5; e < 10; e++)
		std::generate_n(std::back_inserter(test_positions), 250'000, PosGen::Random_with_empty_count(e /*empty_count*/, 113));

	std::cout << "Generated" << std::endl;

	const int64_t train_size = static_cast<int64_t>(train_positions.size());
	Vector train_scores(train_size);
	#pragma omp parallel for schedule(dynamic, 64)
	for (int64_t i = 0; i < train_size; i++)
		train_scores[i] = pvs.Eval(train_positions[i]).window.lower();

	const int64_t test_size = static_cast<int64_t>(test_positions.size());
	Vector test_scores(test_size);
	#pragma omp parallel for schedule(dynamic, 64)
	for (int64_t i = 0; i < test_size; i++)
		test_scores[i] = pvs.Eval(test_positions[i]).window.lower();

	std::cout << "Solved" << std::endl;

	std::vector<std::vector<BitBoard>> ppp = {
		std::vector<BitBoard>{L0, L1, L2, L3, D5, D6, D7, Comet, B5, C4}, // 6.32
		std::vector<BitBoard>{L0, L1, L2, L3, D5, D6, D7, Comet, B5, C4, Q1, Q2}, // 6.21
		std::vector<BitBoard>{L0, L1, L2, L3, D5, D6, D7, Comet, B5, Q0, Q1, Q2},
	};

	for (const auto& patterns : ppp)
	{
		const auto start = std::chrono::high_resolution_clock::now();
		auto config_indexer = CreateConfigIndexer(patterns);
		auto train_mat = CreateMatrix(*config_indexer, train_positions);
		auto test_mat = CreateMatrix(*config_indexer, test_positions);

		Vector weights(config_indexer->reduced_size, 0);

		DiagonalPreconditioner P(train_mat.JacobiPreconditionerSquare(1000));
		PCG solver(transposed(train_mat) * train_mat, P, weights, transposed(train_mat) * train_scores);
		solver.Iterate(10);
		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		const auto milliseconds = duration.count();
		std::cout << milliseconds << "ms. Reduced size: " << config_indexer->reduced_size
			<< "\tTrainError: " << SampleStandardDeviation(train_scores - train_mat * solver.GetX())
			<< "\t TestError: " << SampleStandardDeviation(test_scores - test_mat * solver.GetX()) << std::endl;
	}

	////LSQR solver(mat, weights, scores);
	//for (int i = 0; i < 100; i++)
	//{
	//	for (int j = 0; j < 100; j++)
	//	{
	//		const auto start = std::chrono::high_resolution_clock::now();
	//		solver.Iterate();
	//		const auto end = std::chrono::high_resolution_clock::now();
	//		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//		const auto milliseconds = duration.count();
	//		std::cout << "Residuum: " << solver.Residuum() 
	//			<< "\t TrainError: " << SampleStandardDeviation(train_scores - train_mat * solver.GetX())
	//			<< "\t TestError: " << SampleStandardDeviation(test_scores - test_mat * solver.GetX())
	//			<< "\t" << milliseconds << std::endl;
	//	}
	//	solver.Reinitialize();
	//	std::cout << "Reinitialized\n";
	//}
	return 0;
}