#include <math.h>
#include <assert.h>
#include "include/particle.hh"

template <size_t d>
Particle<d>::Particle(double a, const std::array<double, d> &position)
{
    assert(a > 0);

    diameter_ = a;
    position_ = position;
}

template <size_t d>
void Particle<d>::adjustPosition(double change, unsigned int dimension)
{
    position_.at(dimension) += change;
}

template <size_t d>
inline double Particle<d>::getSquaredDistance()
{
    double sum;
    for (size_t i = 0; i < d; i++)
    {
        sum += position_[i] * position_[i];
    }
    return sum;
}

template <size_t d>
double Particle<d>::distanceTo(std::unique_ptr<Particle<d>> &other)
{
    double sum;
    for (size_t i = 0; i < d; i++)
    {
        double between = position_[i] - other->getPostition()[i];
        sum += between * between;
    }
    return sqrt(sum);
}
