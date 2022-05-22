#pragma once
#include "Pattern/Pattern.h"
#include "IO/IO.h"
#include <istream>
#include <ostream>

// GLEM
void Serialize(const GLEM&, std::ostream&);
template <>
GLEM Deserialize<GLEM>(std::istream&);

// AM
void Serialize(const AM&, std::ostream&);
template <>
AM Deserialize<AM>(std::istream&);

// AAGLEM
void Serialize(const AAGLEM&, std::ostream&);
template <>
AAGLEM Deserialize<AAGLEM>(std::istream&);
