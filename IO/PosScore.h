#pragma once
#include "Core/Core.h"
#include "Stream.h"

//#pragma pack(1)
//struct PosScore // TODO: Remove!
//{
//	Position pos{};
//	int8_t score = undefined_score;
//};
//#pragma pack()
//
//#pragma pack(1)
//struct CompactPosScore
//{
//	Position pos;
//	int8_t score;
//
//	CompactPosScore() noexcept = default;
//	CompactPosScore(PosScore ps) noexcept : pos(ps.pos), score(ps.score) {}
//	operator PosScore() const { return { pos, score }; }
//};
//#pragma pack()
//
//template <>
//struct compact<PosScore>
//{
//	using type = typename CompactPosScore;
//};