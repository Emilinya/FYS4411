#pragma once

#include <array>
#include <memory>
#include <cmath>
#include <assert.h>

template <size_t d>
class Particle
{
public:
    Particle()
    {
        diameter_ = 0;
    }

    Particle(double diameter, const std::array<double, d> &position)
    {
        assert(diameter >= 0);

        diameter_ = diameter;
        position_ = position;
    }

    void adjustPosition(double change, size_t dimension);
    double distanceTo(const Particle<d> &other) const;

    inline const std::array<double, d> &getPosition() const { return position_; }
    inline double getDiameter() const { return diameter_; }
    inline double getSquaredDistance() const;

private:
    std::array<double, d> position_;
    double diameter_;
};

template <size_t d>
void Particle<d>::adjustPosition(double change, size_t dimension)
{
    position_[dimension] += change;
}

template <size_t d>
inline double Particle<d>::getSquaredDistance() const
{
    double sum;
    for (size_t i = 0; i < d; i++)
    {
        sum += position_[i] * position_[i];
    }
    return sum;
}

template <size_t d>
double Particle<d>::distanceTo(const Particle<d> &other) const
{
    double sum;
    for (size_t i = 0; i < d; i++)
    {
        double between = position_[i] - other.getPosition()[i];
        sum += between * between;
    }
    return std::sqrt(sum);
}
