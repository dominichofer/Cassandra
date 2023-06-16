#pragma once
#include "Core/Core.h"
#include <string>
#include <string_view>

// Depth
int DepthFromString(std::string_view depth_at_confidence_level);

// Confidence level
float ConfidenceLevelFromString(std::string_view depth_at_confidence_level);

// Field
bool IsField(std::string_view);
Field FieldFromString(std::string_view);

// Score
bool IsScore(std::string_view);
std::string ScoreToString(int);
int ScoreFromString(std::string_view);

// Position
bool IsPosition(std::string_view);

// PosScore
bool IsPositionScore(std::string_view);
std::string to_string(const PosScore&);
PosScore PosScoreFromString(std::string_view);

// Game
bool IsGame(std::string_view);
std::string to_string(const Game&);
Game GameFromString(std::string_view);

// GameScore
bool IsGameScore(std::string_view);
std::string to_string(const GameScore&);
GameScore GameScoreFromString(std::string_view);

// std::chrono
std::string RichTimeFormat(std::chrono::duration<double>);
