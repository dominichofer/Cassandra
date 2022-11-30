#pragma once
#include "IO/IO.h"
#include <istream>
#include <ostream>

// NoMovePuzzle::Task
void Serialize(const NoMovePuzzle::Task&, std::ostream&);
template <>
NoMovePuzzle::Task Deserialize<NoMovePuzzle::Task>(std::istream&);

// NoMovePuzzle
void Serialize(const NoMovePuzzle&, std::ostream&);
template <>
NoMovePuzzle Deserialize<NoMovePuzzle>(std::istream&);

// AllMovePuzzle::Task::SubTask
void Serialize(const AllMovePuzzle::Task::SubTask&, std::ostream&);
template <>
AllMovePuzzle::Task::SubTask Deserialize<AllMovePuzzle::Task::SubTask>(std::istream&);

// AllMovePuzzle::Task
void Serialize(const AllMovePuzzle::Task&, std::ostream&);
template <>
AllMovePuzzle::Task Deserialize<AllMovePuzzle::Task>(std::istream&);

// AllMovePuzzle
void Serialize(const AllMovePuzzle&, std::ostream&);
template <>
AllMovePuzzle Deserialize<AllMovePuzzle>(std::istream&);
