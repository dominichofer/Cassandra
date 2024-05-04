#include "pch.h"

TEST(EmptyCountFiltered, Position)
{
    std::vector<Position> pos = {
		Position{ 0, 0 },
		Position{ 0, 1 },
		Position{ 0, 3 }
	};
	int empty_count = 63;

	std::vector<Position> result = EmptyCountFiltered(pos, empty_count);

	std::vector<Position> reference = { pos[1] };
	EXPECT_EQ(result, reference);
}

TEST(EmptyCountFiltered, Position_range)
{
	std::vector<Position> pos = {
		Position{ 0, 0 },
		Position{ 0, 1 },
		Position{ 0, 3 }
	};
	int lower = 62;
	int upper = 63;

	std::vector<Position> result = EmptyCountFiltered(pos, lower, upper);

	std::vector<Position> reference = { pos[1], pos[2] };
	EXPECT_EQ(result, reference);
}

TEST(EmptyCountFiltered, ScoredPosition)
{
    std::vector<ScoredPosition> pos_score = {
        ScoredPosition{ Position{ 0, 0 }, +1 },
        ScoredPosition{ Position{ 0, 1 }, -1 },
        ScoredPosition{ Position{ 0, 3 }, +1 }
    };
    int empty_count = 63;

    std::vector<ScoredPosition> result = EmptyCountFiltered(pos_score, empty_count);

    std::vector<ScoredPosition> reference = { pos_score[1] };
    EXPECT_EQ(result, reference);
}

TEST(EmptyCountFiltered, ScoredPosition_range)
{
    std::vector<ScoredPosition> pos_score = {
        ScoredPosition{ Position{ 0, 0 }, +1 },
        ScoredPosition{ Position{ 0, 1 }, -1 },
        ScoredPosition{ Position{ 0, 3 }, +1 }
    };
	int lower = 62;
	int upper = 63;

	std::vector<ScoredPosition> result = EmptyCountFiltered(pos_score, lower, upper);

    std::vector<ScoredPosition> reference = { pos_score[1], pos_score[2] };
    EXPECT_EQ(result, reference);
}
