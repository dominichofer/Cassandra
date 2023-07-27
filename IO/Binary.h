#pragma once
#include "Pattern/Pattern.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

template<class, template<class...> class>
inline constexpr bool is_specialization = false;

template<template<class...> class T, class... Args>
inline constexpr bool is_specialization<T<Args...>, T> = true;

template <typename T>
T Deserialize(std::istream&);

// File
template <typename T>
void Serialize(const T& t, const std::filesystem::path& file)
{
	std::fstream stream(file, std::ios::binary | std::ios::out);
	Serialize(t, stream);
}

template <typename T>
T Deserialize(const std::filesystem::path& file)
{
	std::fstream stream(file, std::ios::binary | std::ios::in);
	return Deserialize<T>(stream);
}

// Arithmetic or Enum
template <typename T>
void Serialize(const T& t, std::ostream& stream)
	requires std::is_arithmetic_v<T> or std::is_enum_v<T>
{
	stream.write(reinterpret_cast<const char*>(std::addressof(t)), sizeof t);
}

template <typename T>
T Deserialize(std::istream& stream)
	requires std::is_arithmetic_v<T> or std::is_enum_v<T>
{
	T t;
	stream.read(reinterpret_cast<char*>(std::addressof(t)), sizeof t);
	return t;
}

// std::vector
template <typename T>
void Serialize(const std::vector<T>& vec, std::ostream& stream)
{
	Serialize(vec.size(), stream);
	for (const T& t: vec)
		Serialize(t, stream);
}

template <typename T>
T Deserialize(std::istream& stream)
	requires is_specialization<T, std::vector>
{
	std::size_t size = Deserialize<std::size_t>(stream);
	T vec;
	vec.reserve(size);
	for (std::size_t i = 0; i < size; i++)
		vec.push_back(Deserialize<typename T::value_type>(stream));
	return vec;
}

// ScoreEstimator
void Serialize(const ScoreEstimator&, std::ostream&);
template <>
ScoreEstimator Deserialize<ScoreEstimator>(std::istream&);

// MSSE
void Serialize(const MultiStageScoreEstimator&, std::ostream&);
template <>
MultiStageScoreEstimator Deserialize<MultiStageScoreEstimator>(std::istream&);

// AccuracyModel
void Serialize(const AccuracyModel&, std::ostream&);
template <>
AccuracyModel Deserialize<AccuracyModel>(std::istream&);

// PatternBasedEstimator
void Serialize(const PatternBasedEstimator&, std::ostream&);
template <>
PatternBasedEstimator Deserialize<PatternBasedEstimator>(std::istream&);

void Save(const PatternBasedEstimator&, const std::filesystem::path&);
PatternBasedEstimator LoadPatternBasedEstimator(const std::filesystem::path&);
