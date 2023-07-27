#pragma once
#include "Board/Board.h"
#include "Hashtable.h"
#include <cstdint>

int64_t Correct(int depth);

class BasicPerft
{
public:
	virtual int64_t calculate(const Position&, int depth);
	virtual int64_t calculate(int depth);

	virtual void clear() {};
};

class UnrolledPerft : public BasicPerft
{
protected:
	const int initial_unroll;
private:
	int64_t calculate_0();
	int64_t calculate_1(const Position&);
	int64_t calculate_2(const Position&);
protected:
	int64_t calculate_n(const Position&, int depth);
public:
	UnrolledPerft(int initial_unroll) : initial_unroll(initial_unroll) {}

	int64_t calculate(const Position&, int depth) override;
	int64_t calculate(int depth) override;
};

class HashTablePerft : public UnrolledPerft
{
protected:
	BigNodeHashTable hash_table;

	int64_t calculate_n(const Position&, int depth);
public:
	HashTablePerft(std::size_t bytes, int initial_unroll);

	int64_t calculate(const Position&, int depth) override;
	int64_t calculate(int depth) override;

	void clear() override { hash_table.clear(); }
};
