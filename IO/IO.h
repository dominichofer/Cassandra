#pragma once
#include "Database.h"
#include "Filebased.h"
#include "Format.h"
#include "Integers.h"
#include "Stream.h"
#include "Table.h"
#include <string>

std::string short_time_format(std::chrono::duration<double> duration);
