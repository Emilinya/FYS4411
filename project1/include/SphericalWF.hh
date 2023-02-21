#pragma once

#include <math.h>
#include <assert.h>
#include "Particle.hh"
#include "WaveFunction.hh"

template <size_t N, size_t d>
class SphericalWF : public WaveFunction<N, d>
{
public:
    SphericalWF(double alpha)
    {
        assert(alpha >= 0);

        alpha_ = alpha;
        this->parameters_ = {alpha};
    }

    double evaluate(const ParticleSystem<N, d> &particleSystem) const;
    double computeLaplacian(const ParticleSystem<N, d> &particleSystem) const;
    double computeLocalEnergy(const ParticleSystem<N, d> &particleSystem) const;
    QForceMat<N, d> computeQForce(const ParticleSystem<N, d> &particleSystem) const;

private:
    double alpha_ = 0;
};

template <size_t N, size_t d>
double SphericalWF<N, d>::evaluate(const ParticleSystem<N, d> &particleSystem) const
{
    double g_prod = std::exp(-alpha_ * particleSystem.getSquareSum());
    double a = particleSystem.getDiameter();

    if (a == 0)
    {
        // a = 0, we don't need to worry about interactions
        return g_prod;
    }

    double f_prod = 1;

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = i + 1; j < N; j++)
        {
            double dist2 = particleSystem[i].squareDistanceTo(particleSystem[j]);

            if (dist2 > a * a)
            {
                f_prod *= 1 - a / std::sqrt(dist2);
            }
            else
            {
                // if any value in a product is 0, the result is 0
                return 0;
            }
        }
    }

    return g_prod * f_prod;
}

template <size_t N, size_t d>
double SphericalWF<N, d>::computeLaplacian(const ParticleSystem<N, d> &particleSystem) const
{
    double const1 = -2. * alpha_ * (double)(d * N);
    double const2 = 4 * alpha_ * alpha_;

    double distSum = particleSystem.getSquareSum();

    return (const1 + const2 * distSum) * evaluate(particleSystem);
}

template <size_t N, size_t d>
double SphericalWF<N, d>::computeLocalEnergy(const ParticleSystem<N, d> &particleSystem) const
{
    double m = 1;
    double omega = 1;

    double const1 = alpha_ * (double)(d * N) / m;
    double const2 = 0.5 * m * omega * omega - 2. * alpha_ * alpha_ / m;

    double distSum = particleSystem.getSquareSum();

    return const1 + const2 * distSum;
}

template <size_t N, size_t d>
QForceMat<N, d> SphericalWF<N, d>::computeQForce(const ParticleSystem<N, d> &particleSystem) const
{
    QForceMat<N, d> qForceMat;
    for (size_t i = 0; i < N; i++)
    {
        auto pos = particleSystem[i].getPosition();
        for (size_t j = 0; j < d; j++)
        {
            qForceMat[i][j] = -4 * alpha_ * pos[j];
        }
    }

    return qForceMat;
}
