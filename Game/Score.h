#pragma once
#include "Board/Board.h"
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>

inline constexpr int min_score{ -32 };
inline constexpr int max_score{ +32 };
inline constexpr int inf_score{ +33 };
inline constexpr int undefined_score{ +35 };

inline constexpr float inf{ std::numeric_limits<float>::infinity() };

CUDA_CALLABLE int EndScore(const Position&) noexcept;

// Depth + Confidence level
std::string DepthClToString(int depth, float confidence_level);
std::tuple<int, float> DepthClFromString(std::string_view);

// Score
bool IsScore(std::string_view);
std::string ScoreToString(int);
int ScoreFromString(std::string_view);