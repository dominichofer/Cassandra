#pragma once
#include "Search/Search.h"
#include "IO/IO.h"
#include <istream>
#include <ostream>

// Confidence
void Serialize(const Confidence&, std::ostream&);
template <>
Confidence Deserialize<Confidence>(std::istream&);

// Intensity
void Serialize(const Intensity&, std::ostream&);
template <>
Intensity Deserialize<Intensity>(std::istream&);
