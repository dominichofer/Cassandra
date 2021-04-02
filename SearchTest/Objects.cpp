#include "pch.h"

namespace Search
{
	const int depth_low = 4;
	const int depth_mid = 5;
	const int depth_high = 6;
	const ConfidenceLevel cl_low(2 /*sigmas*/);
	const ConfidenceLevel cl_mid(3 /*sigmas*/);
	const ConfidenceLevel cl_high(4 /*sigmas*/);
	const Intensity ll{depth_low, cl_low};
	const Intensity lm{depth_low, cl_mid};
	const Intensity lh{depth_low, cl_high};
	const Intensity ml{depth_mid, cl_low};
	const Intensity mm{depth_mid, cl_mid};
	const Intensity mh{depth_mid, cl_high};
	const Intensity hl{depth_high, cl_low};
	const Intensity hm{depth_high, cl_mid};
	const Intensity hh{depth_high, cl_high};

	TEST(Intensity_compare, equal)
	{
		EXPECT_FALSE(mm == ll); EXPECT_FALSE(mm == ml); EXPECT_FALSE(mm == hl);
		EXPECT_FALSE(mm == lm); EXPECT_TRUE (mm == mm); EXPECT_FALSE(mm == hm);
		EXPECT_FALSE(mm == lh); EXPECT_FALSE(mm == mh); EXPECT_FALSE(mm == hh);
	}

	TEST(Intensity_compare, not_equal)
	{
		EXPECT_TRUE (mm != ll); EXPECT_TRUE (mm != ml); EXPECT_TRUE (mm != hl);
		EXPECT_TRUE (mm != lm); EXPECT_FALSE(mm != mm); EXPECT_TRUE (mm != hm);
		EXPECT_TRUE (mm != lh); EXPECT_TRUE (mm != mh); EXPECT_TRUE (mm != hh);
	}

	TEST(Intensity_compare, less)
	{
		EXPECT_FALSE(mm < ll); EXPECT_FALSE(mm < ml); EXPECT_FALSE(mm < hl);
		EXPECT_FALSE(mm < lm); EXPECT_FALSE(mm < mm); EXPECT_TRUE (mm < hm);
		EXPECT_FALSE(mm < lh); EXPECT_TRUE (mm < mh); EXPECT_TRUE (mm < hh);
	}

	TEST(Intensity_compare, less_equal)
	{
		EXPECT_FALSE(mm <= ll); EXPECT_FALSE(mm <= ml); EXPECT_FALSE(mm <= hl);
		EXPECT_FALSE(mm <= lm); EXPECT_TRUE (mm <= mm); EXPECT_TRUE (mm <= hm);
		EXPECT_FALSE(mm <= lh); EXPECT_TRUE (mm <= mh); EXPECT_TRUE (mm <= hh);
	}

	TEST(Intensity_compare, greater)
	{
		EXPECT_TRUE (mm > ll); EXPECT_TRUE (mm > ml); EXPECT_FALSE(mm > hl);
		EXPECT_TRUE (mm > lm); EXPECT_FALSE(mm > mm); EXPECT_FALSE(mm > hm);
		EXPECT_FALSE(mm > lh); EXPECT_FALSE(mm > mh); EXPECT_FALSE(mm > hh);
	}

	TEST(Intensity_compare, greater_equal)
	{
		EXPECT_TRUE (mm >= ll); EXPECT_TRUE (mm >= ml); EXPECT_FALSE(mm >= hl);
		EXPECT_TRUE (mm >= lm); EXPECT_TRUE (mm >= mm); EXPECT_FALSE(mm >= hm);
		EXPECT_FALSE(mm >= lh); EXPECT_FALSE(mm >= mh); EXPECT_FALSE(mm >= hh);
	}
}