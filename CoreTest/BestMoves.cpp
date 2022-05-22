#pragma once
#include "pch.h"

namespace BestMovesTest
{
	BestMoves Added(BestMoves reseiver, auto novum)
	{
		reseiver.Add(novum);
		return reseiver;
	}

	Field XX = Field::invalid;
	Field A1 = Field::A1;
	Field A2 = Field::A2;
	Field B1 = Field::B1;
	Field B2 = Field::B2;
	BestMoves XX_XX{ XX, XX };
	BestMoves A1_XX{ A1, XX };
	BestMoves A2_XX{ A2, XX };
	BestMoves A1_A2{ A1, A2 };
	BestMoves B1_XX{ B1, XX };
	BestMoves B1_B2{ B1, B2 };
	BestMoves B1_A1{ B1, A1 };
	BestMoves A2_A1{ A2, A1 };
	BestMoves A1_B1{ A1, B1 };
	BestMoves A2_B1{ A2, B1 };
	BestMoves B1_A2{ B1, A2 };

	TEST(add_one, XX_XX_add_XX_is_XX_XX) { EXPECT_EQ(Added(XX_XX, XX), XX_XX); }
	TEST(add_one, XX_XX_add_A1_is_A1_XX) { EXPECT_EQ(Added(XX_XX, A1), A1_XX); }
	TEST(add_one, XX_XX_add_A2_is_A2_XX) { EXPECT_EQ(Added(XX_XX, A2), A2_XX); }
	TEST(add_one, XX_XX_add_B1_is_B1_XX) { EXPECT_EQ(Added(XX_XX, B1), B1_XX); }

	TEST(add_one, A1_XX_add_XX_is_A1_XX) { EXPECT_EQ(Added(A1_XX, XX), A1_XX); }
	TEST(add_one, A1_XX_add_A1_is_A1_XX) { EXPECT_EQ(Added(A1_XX, A1), A1_XX); }
	TEST(add_one, A1_XX_add_A2_is_A2_A1) { EXPECT_EQ(Added(A1_XX, A2), A2_A1); }
	TEST(add_one, A1_XX_add_B1_is_B1_A1) { EXPECT_EQ(Added(A1_XX, B1), B1_A1); }

	TEST(add_one, A1_A2_add_XX_is_A1_A2) { EXPECT_EQ(Added(A1_A2, XX), A1_A2); }
	TEST(add_one, A1_A2_add_A1_is_A1_A2) { EXPECT_EQ(Added(A1_A2, A1), A1_A2); }
	TEST(add_one, A1_A2_add_A2_is_A2_A1) { EXPECT_EQ(Added(A1_A2, A2), A2_A1); }
	TEST(add_one, A1_A2_add_B1_is_B1_A1) { EXPECT_EQ(Added(A1_A2, B1), B1_A1); }

	TEST(add_two, XX_XX_add_XX_XX_is_XX_XX) { EXPECT_EQ(Added(XX_XX, XX_XX), XX_XX); }
	TEST(add_two, XX_XX_add_A1_XX_is_A1_XX) { EXPECT_EQ(Added(XX_XX, A1_XX), A1_XX); }
	TEST(add_two, XX_XX_add_A1_A2_is_A1_A2) { EXPECT_EQ(Added(XX_XX, A1_A2), A1_A2); }

	TEST(add_two, A1_XX_add_XX_XX_is_A1_XX) { EXPECT_EQ(Added(A1_XX, XX_XX), A1_XX); }
	TEST(add_two, A1_XX_add_B1_XX_is_B1_A1) { EXPECT_EQ(Added(A1_XX, B1_XX), B1_A1); }
	TEST(add_two, A1_XX_add_B1_B2_is_B1_A1) { EXPECT_EQ(Added(A1_XX, B1_B2), B1_A1); }
	TEST(add_two, A1_XX_add_A1_B1_is_A1_B1) { EXPECT_EQ(Added(A1_XX, A1_B1), A1_B1); }
	TEST(add_two, A1_XX_add_B1_A1_is_B1_A1) { EXPECT_EQ(Added(A1_XX, B1_A1), B1_A1); }

	TEST(add_two, A1_A2_add_XX_XX_is_A1_A2) { EXPECT_EQ(Added(A1_A2, XX_XX), A1_A2); }
	TEST(add_two, A1_A2_add_A1_A2_is_A1_A2) { EXPECT_EQ(Added(A1_A2, A1_A2), A1_A2); }
	TEST(add_two, A1_A2_add_A1_B1_is_A1_B1) { EXPECT_EQ(Added(A1_A2, A1_B1), A1_B1); }
	TEST(add_two, A1_A2_add_A2_A1_is_A2_A1) { EXPECT_EQ(Added(A1_A2, A2_A1), A2_A1); }
	TEST(add_two, A1_A2_add_A2_B1_is_A2_A1) { EXPECT_EQ(Added(A1_A2, A2_B1), A2_A1); }
	TEST(add_two, A1_A2_add_B1_XX_is_B1_A1) { EXPECT_EQ(Added(A1_A2, B1_XX), B1_A1); }
	TEST(add_two, A1_A2_add_B1_A1_is_B1_A1) { EXPECT_EQ(Added(A1_A2, B1_A1), B1_A1); }
	TEST(add_two, A1_A2_add_B1_A2_is_B1_A1) { EXPECT_EQ(Added(A1_A2, B1_A2), B1_A1); }
	TEST(add_two, A1_A2_add_B1_B2_is_B1_A1) { EXPECT_EQ(Added(A1_A2, B1_B2), B1_A1); }
}