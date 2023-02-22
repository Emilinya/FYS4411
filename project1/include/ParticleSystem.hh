#pragma once

#include <memory>
#include <vector>
#include <assert.h>

#include "utils.hh"
#include "Random.hh"
#include "Particle.hh"

template <size_t N, size_t d>
class ParticleSystem
{
public:
    ParticleSystem(double a, double m, double boxSize) : m_(m), a_(a)
    {
        assert(N > 0 && d > 0);
        assert(a >= 0 && m >= 0);

        Random random;

        for (size_t i = 0; i < N; i++)
        {
            std::array<double, d> position;

            for (size_t j = 0; j < d; j++)
            {
                position[j] = random.nextDouble(0, boxSize);
            }

            particles_[i] = Particle<d>(position);
        }
    }

    double getSquareSum() const
    {
        double sum = 0;
        for (size_t i = 0; i < N; i++)
        {
            sum += particles_[i].getSquaredDistance();
        }
        return sum;
    }

    inline double getDiameter() const
    {
        return a_;
    }

    inline double getMass() const
    {
        return m_;
    }

    inline void setAt(size_t idx, const Particle<d> &other) {
        particles_[idx] = other;
    }

    inline const Particle<d> &operator[](size_t idx) const
    {
        return particles_[idx];
    }

private:
    std::array<Particle<d>, N> particles_;
    double m_;
    double a_;
};
