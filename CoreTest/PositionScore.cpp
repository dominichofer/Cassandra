#include "pch.h"

TEST(PosScore, to_string)
{
	PosScore ps{ Position::Start(), +13 };
	ASSERT_EQ(to_string(ps), "---------------------------OX------XO--------------------------- X % +26");
}