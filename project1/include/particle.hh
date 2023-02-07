#pragma once

#include <array>
#include <memory>

template <size_t d>
class Particle
{
public:
    Particle(double diameter, const std::array<double, d> &position);

    void adjustPosition(double change, unsigned int dimension);
    double distanceTo(std::unique_ptr<Particle<d>> &other);

    inline std::array<double, d> &getPosition() { return position_; }
    inline double getDiameter() { return diameter_; }
    inline double getSquaredDistance();

private:
    std::array<double, d> position_;
    double diameter_;
};
