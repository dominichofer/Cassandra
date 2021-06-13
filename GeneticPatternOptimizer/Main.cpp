#include "Core/Core.h"
#include "IO/IO.h"
#include "Pattern/DenseIndexer.h"
#include "Math/DenseMatrix.h"
#include "Math/MatrixCSR.h"
#include "Math/Vector.h"
#include "Math/Solver.h"
#include "Math/Statistics.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>
#include <string>
#include <chrono>
#include <array>
#include <random>
#include <type_traits>

class Entity;
using Probability = double;

double fitness_eval(const Entity& ); // forward declaration

template <typename T>
T rnd(T min, T max)
{
    static std::mt19937_64 rng(std::random_device{}()); // random number generator

    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng);
    }
    else
    {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(rng);
    }
}

template <typename T>
T rnd();

template <>
Probability rnd<Probability>() { return rnd<Probability>(0, 1); }

template <>
bool rnd<bool>() { return rnd<int>(0, 1) == 0; }

class Activator
{
    std::vector<bool> array;

    std::vector<bool>::reference choose_one(const std::vector<std::vector<bool>::reference>& options)
    {
        assert(!options.empty());
        return options[rnd<int>(0, options.size() - 1)];
    }

    std::vector<std::vector<bool>::reference> in_state(bool state)
    {
        std::vector<std::vector<bool>::reference> ret;
        for (std::vector<bool>::reference e : array)
            if (e == state)
                ret.push_back(e);
        return ret;
    }

    void activate_one() { choose_one(in_state(false)) = true; }
    void deactivate_one() { choose_one(in_state(true)) = false; }

    Activator& randomize(int min_active, int max_active)
    {
        for (int i = rnd<int>(min_active, max_active); i > 0; i--)
            activate_one();
        return *this;
    }
public:
    Activator(int size = 0) : array(size, false) {}
    Activator(std::vector<bool> v) : array(std::move(v)) {}
    static Activator Random(int size, int min_active, int max_active)
    {
        return Activator{size}.randomize(min_active, max_active);
    }
    void mutate()
    {
        if ((rnd<Probability>() < 0.75) && (in_state(true).size() > 1)) // deactivate one. 75% chance.
            deactivate_one();
        if ((rnd<Probability>() < 0.75) && (!in_state(false).empty())) // activate one. 75% chance.
            activate_one();
    }
    auto Array() const { return array; }
    auto operator[](int index) const { return array[index]; }
};

class Actor
{
    bool is_legal() const { return (popcount(bb) <= upper_limit()) && (popcount(bb) > 0); }
    static BitBoard choose_one(BitBoard available)
    {
        assert(available != BitBoard{0});
        auto index = rnd<int>(0, popcount(available) - 1);
        return PDep(1ULL << index, available);
    }
    void add_one() { bb ^= Expand(choose_one(UniqueBits(~bb))); }
    void remove_one() { bb ^= Expand(choose_one(UniqueBits(bb))); }
protected:
    BitBoard bb = 0;
    virtual int upper_limit() const = 0;
    virtual BitBoard UniqueBits(BitBoard) const = 0;
    virtual BitBoard Expand(BitBoard) const = 0;
    void randomize()
    {
        bb = 0;
        add_one();
        for (int i = 0; i < 6; i++)
            if (rnd<bool>())
                add_one();
    }
public:
    Actor() = default;
    Actor(BitBoard bb) : bb(bb) {}
    virtual std::unique_ptr<Actor> clone() const = 0;
    std::unique_ptr<Actor> UniformCrossover(const Actor& o) const
    {
        auto ret = o.clone();
        auto mask = Expand(UniqueBits(rnd<uint64_t>(0, -1)));
        ret->bb = (ret->bb & mask) | (bb & ~mask);
        while (popcount(ret->bb) > ret->upper_limit())
            ret->remove_one();
        while (!ret->bb)
            ret->add_one();
        return ret;
    }
    void mutate()
    {
        if ((rnd<Probability>() < 0.75) && (popcount(bb) > 1)) // remove one. 75% chance.
            remove_one();
        if ((rnd<Probability>() < 0.75) && (popcount(bb) < upper_limit())) // add one. 75% chance.
            add_one();
        if ((rnd<Probability>() < 0.25) && (popcount(bb) < upper_limit())) // add one. 25% chance.
            add_one();
        if ((rnd<Probability>() < 0.25) && (popcount(bb) > 1)) // remove one. 25% chance.
            remove_one();
        assert(is_legal());
    }
    BitBoard Phenotype() const { return bb; }
    virtual int Cost() const = 0;
};

class AsymmetricActor final : public Actor
{
    int upper_limit() const override { return 12; }
    BitBoard UniqueBits(BitBoard b) const override { return b; }
    BitBoard Expand(BitBoard b) const override { return b; }
public:
    AsymmetricActor() = default;
    AsymmetricActor(BitBoard bb) : Actor(bb) {}
    static AsymmetricActor Random() { AsymmetricActor x; x.randomize(); return x; }
    std::unique_ptr<Actor> clone() const override { return std::make_unique<AsymmetricActor>(bb); }
    int Cost() const override { return 8; }
};

class HorizontalActor final : public Actor
{
    int upper_limit() const override { return 12; }
    BitBoard UniqueBits(BitBoard b) const override { return b & 0x0F0F0F0F0F0F0F0FULL; }
    BitBoard Expand(BitBoard b) const override { return b | FlipHorizontal(b); }
public:
    HorizontalActor() = default;
    HorizontalActor(BitBoard bb) : Actor(bb) {}
    static HorizontalActor Random() { HorizontalActor x; x.randomize(); return x; }
    std::unique_ptr<Actor> clone() const override { return std::make_unique<HorizontalActor>(bb); }
    int Cost() const override { return 4; }
};

class DiagonalActor final : public Actor
{
    int upper_limit() const override { return 16; }
    BitBoard UniqueBits(BitBoard b) const override { return b & 0xFF7F3F1F0F070301ULL; }
    BitBoard Expand(BitBoard b) const override { return b | FlipDiagonal(b); }
public:
    DiagonalActor() = default;
    DiagonalActor(BitBoard bb) : Actor(bb) {}
    static DiagonalActor Random() { DiagonalActor x; x.randomize(); return x; }
    std::unique_ptr<Actor> clone() const override { return std::make_unique<DiagonalActor>(bb); }
    int Cost() const override { return 4; }
};

class Chromosome
{
public:
    Activator activator;
    std::vector<std::unique_ptr<Actor>> actors;
    Chromosome() = default;
    Chromosome(const Chromosome& o) : activator(o.activator)
    {
        std::transform(o.actors.begin(), o.actors.end(), std::back_inserter(actors), [](const auto& x) { return x->clone(); });
    }
    Chromosome(Chromosome&&) noexcept = default;
    Chromosome& operator=(const Chromosome& o)
    {
        if (&o == this)
            return *this;
        activator = o.activator;
        actors.clear();
        std::transform(o.actors.begin(), o.actors.end(), std::back_inserter(actors), [](const auto& x) { return x->clone(); });
    }
    Chromosome& operator=(Chromosome&&) noexcept = default;

    static Chromosome Random(int as, int hs, int ds, int min_active, int max_active)
    {
        Chromosome ret;
        ret.activator = Activator::Random(as + hs + ds, min_active, max_active);
        for (int i = 0; i < as; i++)
            ret.actors.emplace_back(std::make_unique<AsymmetricActor>(AsymmetricActor::Random()));
        for (int i = 0; i < hs; i++)
            ret.actors.emplace_back(std::make_unique<HorizontalActor>(HorizontalActor::Random()));
        for (int i = 0; i < ds; i++)
            ret.actors.emplace_back(std::make_unique<DiagonalActor>(DiagonalActor::Random()));
        return ret;
    }
    Chromosome UniformCrossover(const Chromosome& o) const
    {
        Chromosome ret;
        ret.activator = rnd<bool>() ? activator : o.activator; // TODO: Maybe this should be UniformCrossover(activator, o.activator).
        for (int i = 0; i < actors.size(); i++)
            ret.actors.emplace_back(actors[i]->UniformCrossover(*o.actors[i]));
        return ret;
    }
    // mutates each gene with a probabiliby of p.
    void mutate(Probability p)
    {
        if (rnd<Probability>() < p / (actors.size() + 1))
            activator.mutate();
        for (auto& gene : actors)
            if (rnd<Probability>() < p)
                gene->mutate();
    }
    auto Activations() const { return activator.Array(); }
    std::vector<BitBoard> Genotype() const
    {
        std::vector<BitBoard> ret;
        for (int i = 0; i < actors.size(); i++)
            ret.push_back(actors[i]->Phenotype());
        return ret;
    }
    std::vector<BitBoard> Phenotype() const
    {
        std::vector<BitBoard> ret;
        for (int i = 0; i < actors.size(); i++)
            if (activator[i])
                ret.push_back(actors[i]->Phenotype());
        return ret;
    }
    int GenoCost() const
    {
        int sum = 0;
        for (int i = 0; i < actors.size(); i++)
            sum += actors[i]->Cost();
        return sum;
    }
    int PhenoCost() const
    {
        int sum = 0;
        for (int i = 0; i < actors.size(); i++)
            if (activator[i])
                sum += actors[i]->Cost();
        return sum;
    }
};

class Entity
{
    Chromosome chr;
    mutable std::optional<double> fitness = std::nullopt;
public:
    Entity(const Entity&) = default;
    Entity(Entity&&) noexcept = default;
    Entity& operator=(const Entity&) = default;
    Entity& operator=(Entity&&) noexcept = default;
    Entity(Chromosome chr) noexcept : chr(std::move(chr)) {}

    static Entity Random(int as, int hs, int ds, int min_active, int max_active)
    {
        return { Chromosome::Random(as, hs, ds, min_active, max_active) };
    }

    bool operator<(const Entity& o) const { return Fitness() < o.Fitness(); }

    void mutate(Probability p) { chr.mutate(p); fitness = std::nullopt; }
    double Fitness() const { if (!fitness) fitness = fitness_eval(*this); return fitness.value(); }
    std::vector<BitBoard> Genotype() const { return chr.Genotype(); }
    std::vector<BitBoard> Phenotype() const { return chr.Phenotype(); }
    int GenoCost() const { return chr.GenoCost(); }
    int PhenoCost() const { return chr.PhenoCost(); }
    Entity Child(const Entity& o) const { return { chr.UniformCrossover(o.chr) }; }

    std::string to_string() const
    {
        std::string ret = "@@@@ Entity @@@\n";
        ret += "fitness: " + std::to_string(Fitness()) + "\n";
        auto active = chr.Activations();
        auto genes = chr.Genotype();
        for (int i = 0; i < genes.size(); i++)
            ret += (active[i] ? "+ " : "- ") + SingleLine(genes[i]) + '\n';
        return ret;
    }
};

std::string to_string(const Entity& e) { return e.to_string(); }

Entity Child(const Entity& l, const Entity& r) { return l.Child(r); }

class Population
{
    std::vector<Entity> entities; // always sorted!

    void add(Entity&& e) { entities.emplace_back(std::move(e)); }
    void sort() { std::sort(entities.begin(), entities.end()); }
public:
    Population() = default;
    Population(std::vector<Entity> entities) : entities(std::move(entities)) {}
    Population(const Population& o) : entities(o.entities) {}
    static Population Random(int popcount, int as, int hs, int ds, int min_active, int max_active)
    {
        std::vector<Entity> vec;
        std::generate_n(std::back_inserter(vec), popcount, [&]() { return Entity::Random(as, hs, ds, min_active, max_active); });
        return { vec };
    }

    auto Entities() const { return entities; }

    void Selection(std::size_t popcount)
    {
        sort();
        entities.erase(entities.begin() + std::min(popcount, size()), entities.end());
    }
    Population Reproduce() const
    {
        Population next;
        const int64_t size = static_cast<int64_t>(entities.size());
        for (int64_t i = 0; i < size; i++)
        {
            auto partner_index = rnd<int>(0, size-1);
            next.add(Child(entities[i], entities[partner_index]));
        }
        return next;
    }
    void Mutation(Probability mutation)
    {
        for (std::size_t i = 1; i < entities.size(); i++)
            entities[i].mutate(mutation);
    }
    void add(Population pop)
    {
        std::copy(pop.entities.begin(), pop.entities.end(), std::back_inserter(entities));
    }
    std::size_t size() const { return entities.size(); }
    std::string to_string()
    {
        sort();
        std::string ret;
        for (int i = 0; i < 4; i ++)
            ret += ::to_string(entities[i]);
        return ret;
    }
};

std::string to_string(Population& pop) { return pop.to_string(); }

constexpr BitBoard L0 = 0x00000000000000FFULL;
constexpr BitBoard L1 = 0x000000000000FF00ULL;
constexpr BitBoard L2 = 0x0000000000FF0000ULL;
constexpr BitBoard L3 = 0x00000000FF000000ULL;
constexpr BitBoard D2 = 0x0000000000000102ULL;
constexpr BitBoard D3 = 0x0000000000010204ULL;
constexpr BitBoard D4 = 0x0000000001020408ULL;
constexpr BitBoard D5 = 0x0000000102040810ULL;
constexpr BitBoard D6 = 0x0000010204081020ULL;
constexpr BitBoard D7 = 0x0001020408102040ULL;
constexpr BitBoard D8 = 0x0102040810204080ULL;
constexpr BitBoard C8 = 0x8040201008040201ULL;
constexpr BitBoard B5 = 0x0000000000001F1FULL;
constexpr BitBoard Q0 = 0x0000000000070707ULL;
constexpr BitBoard Q1 = 0x0000000000070707ULL << 9;
constexpr BitBoard Q2 = 0x0000000000070707ULL << 18;
constexpr BitBoard Ep = 0x0000000000003CBDULL;
constexpr BitBoard Epp = 0x0000000000003CFFULL;
constexpr BitBoard C3 = 0x0000000000010307ULL;
constexpr BitBoard C4 = 0x000000000103070FULL;
constexpr BitBoard C4p = 0x000000000107070FULL;
constexpr BitBoard C3p1 = 0x000000000101030FULL;
constexpr BitBoard C3p2 = 0x000000010101031FULL;
constexpr BitBoard C3p3 = 0x000001010101033FULL;
constexpr BitBoard C4p1 = 0x000000010103071FULL;
constexpr BitBoard Comet = 0x8040201008040303ULL;
constexpr BitBoard Cometp = 0xC0C0201008040303ULL;
constexpr BitBoard C3pp = 0x81010000000103C7ULL;
constexpr BitBoard C3ppp = 0x81410000000103C7ULL;

constexpr BitBoard C4pp = C4 | C3pp;
constexpr BitBoard AA = 0x000000010105031FULL;


auto CreateMatrix(const DenseIndexer& indexer, const std::vector<Position>& pos)
{
    auto row_size = indexer.variations;
    auto cols = indexer.reduced_size;
    auto rows = pos.size();
    MatrixCSR<uint32_t> mat(row_size, cols, rows);

    const int64 size = pos.size();
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++)
        for (int j = 0; j < indexer.variations; j++)
            mat.begin(i)[j] = indexer.DenseIndex(pos[i], j);
    return mat;
}

std::vector<Position> train_positions;
std::vector<Position> test_positions;
Vector train_scores, test_scores;

double fitness_eval(const Entity& entity)
{
    auto phenotype = entity.Phenotype();
    auto indexer = CreateDenseIndexer(phenotype);
    auto train_mat = CreateMatrix(*indexer, train_positions);
    auto test_mat = CreateMatrix(*indexer, test_positions);
    Vector weights(indexer->reduced_size, 0);

    DiagonalPreconditioner P(train_mat.JacobiPreconditionerSquare(100));
    PCG solver(transposed(train_mat) * train_mat, P, weights, transposed(train_mat) * train_scores);
    solver.Iterate(10);
    double fit = StandardDeviation(test_scores - test_mat * solver.X());

    return fit + std::max(0, entity.PhenoCost()-32);
}

int main(int argc, char *argv[])
{
    //auto indexer = CreateDenseIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, Ep, C3p1, B5 }); // 6.38253
    //auto indexer = CreateDenseIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, Epp, C3p1, B5 }); // 6.46631
    //auto indexer = CreateDenseIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, C3p1, B5 }); //6.38712
    //auto indexer = CreateDenseIndexer({ L02X, L1, L2, L3, D4, D5, D6, D7, Comet, C3p1, B6 }); // 6.5572

    //HashTablePVS tt{1'000'000};
    //Search::PV pvs{tt};

    for (int i = 18; i <= 22; i++)
    {
        const auto input = R"(G:\Reversi\rnd\e)" + std::to_string(i) + ".psc";
        std::vector<PosScore> data = LoadVec_old<PosScore>(input);
        for (int i = 0; i < data.size(); i++)
        {
        	if (i < 800'000)
        	{
        		test_positions.push_back(data[i].pos);
        		test_scores.push_back(data[i].score);
        	}
        	else
        	{
        		train_positions.push_back(data[i].pos);
        		train_scores.push_back(data[i].score);
        	}
        }
    }

    //for (int e = 5; e < 10; e++)
    //    std::generate_n(std::back_inserter(train_positions), 1'000'000, PosGen::RandomWithEmptyCount(e /*empty_count*/, 13));
    //for (int e = 5; e < 10; e++)
    //    std::generate_n(std::back_inserter(test_positions), 250'000, PosGen::RandomWithEmptyCount(e /*empty_count*/, 113));

    std::cout << "Generated" << std::endl;

    //const int64_t train_size = static_cast<int64_t>(train_positions.size());
    //train_scores = Vector(train_size);
    //#pragma omp parallel for
    //for (int64_t i = 0; i < train_size; i++)
    //    train_scores[i] = pvs.Eval(train_positions[i]).window.lower();

    //const int64_t test_size = static_cast<int64_t>(test_positions.size());
    //test_scores = Vector(test_size);
    //#pragma omp parallel for
    //for (int64_t i = 0; i < test_size; i++)
    //    test_scores[i] = pvs.Eval(test_positions[i]).window.lower();

    //std::cout << "Solved" << std::endl;

    Population pop;
    Chromosome chr;
    chr.activator = Activator({true, true, true, true, true, false, true, true});
    chr.actors.emplace_back(std::make_unique<AsymmetricActor>(B5));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- # - - - - # -"
        "# # # # # # # #"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "# # # # # # # #"
        "- # - - - - # -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L2));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L3));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - # # - - -"
        "- - # # # # - -"
        "- - - # # - - -"
        "- - - - - - - -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(C4));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(
        "# # - - - - - -"
        "# # # - - - - -"
        "- # # # - - - -"
        "- - # # - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_BitBoard));
    pop.add(Population(std::vector<Entity>{Entity(chr)}));

    chr.actors.clear();
    chr.actors.emplace_back(std::make_unique<AsymmetricActor>(B5));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - # # # # - -"
        "- - - # # - - -"
        "- - - # # - - -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "# - - - - - - #"
        "# # - - - - # #"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- # - - - - # -"
        "- - - - - - - -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- # # - - # # -"
        "# # - - - - # #"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- # - - - - # -"
        "# # # # # # # #"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - # # - - -"
        "- - # # # # - -"
        "- - - # # - - -"
        "- - - - - - - -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(C4));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(
        "# # - - - - - -"
        "# # # - - - - -"
        "- # # # - - - -"
        "- - # # - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_BitBoard));
    pop.add(Population(std::vector<Entity>{Entity(chr)}));

    chr.actors.clear();
    chr.actors.emplace_back(std::make_unique<AsymmetricActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - # - -"
        "- - - # # # # -"
        "- - - # # # # #"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - # # - - -"
        "- - - - - - - -"
        "# # # # # # # #"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "# # # # # # # #"
        "- # - - - - # -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L2));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L3));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - # # - - -"
        "- - # # # # - -"
        "- - - # # - - -"
        "- - - - - - - -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(C4));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(
        "# # - - - - - -"
        "# # # - - - - -"
        "- # # # - - - -"
        "- - # # - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_BitBoard));
    pop.add(Population(std::vector<Entity>{Entity(chr)}));

    chr.actors.clear();
    chr.actors.emplace_back(std::make_unique<AsymmetricActor>(B5));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - # # - - -"
        "- - - - - - - -"
        "# # # # # # # #"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "# # # # # # # #"
        "- # - - - - # -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L2));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L3));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - # # - - -"
        "- - # # # # - -"
        "- - - # # - - -"
        "- - - - - - - -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(C4));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(
        "# # - - - - - -"
        "# # # - - - - -"
        "- # # # - - - -"
        "- - # # - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_BitBoard));
    pop.add(Population(std::vector<Entity>{Entity(chr)}));

    chr.actors.clear();
    chr.actors.emplace_back(std::make_unique<AsymmetricActor>(B5));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L0));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L1));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L2));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(L3));
    chr.actors.emplace_back(std::make_unique<HorizontalActor>(
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - # # - - -"
        "- - # # # # - -"
        "- - - # # - - -"
        "- - - - - - - -"_BitBoard));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(C4));
    chr.actors.emplace_back(std::make_unique<DiagonalActor>(
        "# # - - - - - -"
        "# # # - - - - -"
        "- # # # - - - -"
        "- - # # - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_BitBoard));
    pop.add(Population(std::vector<Entity>{Entity(chr)}));
    pop.add(Population::Random(20, 1 /*asymmetrics*/, 5 /*horizontals*/, 2 /*diagonals*/, 8, 8));
    for (int i = 0; i < 10000; i++)
    {
        std::cout << "Generation " << i << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
            << to_string(pop) << "\n";

        Population next = pop.Reproduce();
        next.Mutation(std::sin(i) * 0.3 + 0.4);
        pop.Selection(15);
        pop.add(next);
        pop.Selection(20);
    }
    return 0;
}

/*
fitness: 6.252528
--------
--------
--------
--------
--------
-----#--
---####-
---#####

--------
--------
#-------
-------#
--------
---##---
--###---
--##----

--------  --------  --------  --------  --------
--------  --------  --------  --------  --------
--------  --------  --------  --------  --------
--------  --------  --------  --------  --------
--------  --------  #------#  #-####-#  ---##---
---##---  --------  #-####-#  --------  --####--
--------  ########  --------  --------  ---##---
########  -#----#-  --------  --------  --------

--------
--------
--------
--------
-------#
------##
-----###
----####

##------
###-----
-###----
--##----
--------
--------
--------
--------
*/

//int main()
//{
//    std::vector<std::vector<BitBoard>> ppp = {
//        //std::vector<BitBoard>{L0, L1, L2, L3, D5, D6, D7, Comet, B5, C4}, // 6.32
//        //std::vector<BitBoard>{L0, L1, L2, L3, D5, D6, D7, Comet, B5, C4, Q1, Q2}, // 6.21
//        //std::vector<BitBoard>{L0, L1, L2, L3, D5, D6, D7, Comet, B5, Q0, Q1, Q2},
//        std::vector<BitBoard>{ // 5.11, 6.11, 6.63, 6.95, 7.18, 7.35, 7.50
//        B5
//            ,
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- # - - - - # -"
//            "# # # # # # # #"_BitBoard
//            ,
//            "# - - - - - - #"
//            "# # - - - - # #"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- # - - - - # -"
//            "- - - - - - - -"_BitBoard
//            ,
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- # # - - # # -"
//            "# # - - - - # #"_BitBoard
//            ,
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - # # # # - -"
//            "- - - # # - - -"
//            "- - - # # - - -"_BitBoard
//            ,C4
//            ,
//            "# # - - - - - -"
//            "# # # - - - - -"
//            "- # # # - - - -"
//            "- - # # - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"
//            "- - - - - - - -"_BitBoard
//    }
//    };
//
//    std::vector<PosScore> data1 = Load<PosScore>(R"(G:\Reversi\rnd\e18.psc)");
//    std::vector<PosScore> data2 = Load<PosScore>(R"(G:\Reversi\rnd\e19.psc)");
//    std::vector<PosScore> data3 = Load<PosScore>(R"(G:\Reversi\rnd\e20.psc)");
//    std::vector<PosScore> data4 = Load<PosScore>(R"(G:\Reversi\rnd\e21.psc)");
//    std::vector<PosScore> data5 = Load<PosScore>(R"(G:\Reversi\rnd\e22.psc)");
//
//    for (int i = 990'000; i < 1'000'000; i++)
//    {
//        test_positions.push_back(data1[i].pos);
//        test_positions.push_back(data2[i].pos);
//        test_positions.push_back(data3[i].pos);
//        test_positions.push_back(data4[i].pos);
//        test_positions.push_back(data5[i].pos);
//        test_scores.push_back(data1[i].score);
//        test_scores.push_back(data2[i].score);
//        test_scores.push_back(data3[i].score);
//        test_scores.push_back(data4[i].score);
//        test_scores.push_back(data5[i].score);
//    }
//
//    for (int j = 1; j < 100; j++)
//    {
//        train_positions.clear();
//        train_scores.clear();
//        for (int i = 0; i < j * 10'000; i++)
//        {
//            train_positions.push_back(data1[i].pos);
//            train_positions.push_back(data2[i].pos);
//            train_positions.push_back(data3[i].pos);
//            train_positions.push_back(data4[i].pos);
//            train_positions.push_back(data5[i].pos);
//            train_scores.push_back(data1[i].score);
//            train_scores.push_back(data2[i].score);
//            train_scores.push_back(data3[i].score);
//            train_scores.push_back(data4[i].score);
//            train_scores.push_back(data5[i].score);
//        }
//
//        for (const auto& patterns : ppp)
//        {
//            const auto start = std::chrono::high_resolution_clock::now();
//            auto indexer = CreateDenseIndexer(patterns);
//            auto train_mat = CreateMatrix(*indexer, train_positions);
//            auto test_mat = CreateMatrix(*indexer, test_positions);
//
//            Vector weights(indexer->reduced_size, 0);
//
//            DiagonalPreconditioner P(train_mat.JacobiPreconditionerSquare(1000));
//            PCG solver(transposed(train_mat) * train_mat, P, weights, transposed(train_mat) * train_scores);
//            solver.Iterate(10);
//            const auto end = std::chrono::high_resolution_clock::now();
//            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//            const auto milliseconds = duration.count();
//            std::cout << milliseconds << "ms. Reduced size: " << indexer->reduced_size
//                << "\tTrainError: " << SampleStandardDeviation(train_scores - train_mat * solver.GetX())
//                << "\t TestError: " << SampleStandardDeviation(test_scores - test_mat * solver.GetX()) << std::endl;
//        }
//    }
//    return 0;
//}