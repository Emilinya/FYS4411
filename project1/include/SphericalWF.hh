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

    double evaluate(const ParticleRay<N, d> &particles);
    double computeLaplacian(const ParticleRay<N, d> &particles);
    double computeLocalEnergy(const ParticleRay<N, d> &particles);
    QForceMat<N, d> computeQForce(const ParticleRay<N, d> &particles);

private:
    double alpha_ = 0;
};

template <size_t N, size_t d>
double SphericalWF<N, d>::evaluate(const ParticleRay<N, d> &particles)
{
    if (N == 1)
    {
        return exp(-alpha_ * particles[0].getSquaredDistance());
    }

    double g_prod;
    double f_prod;

    for (size_t i = 0; i < N; i++)
    {
        g_prod *= exp(-alpha_ * particles[i].getSquaredDistance());

        double ai = particles[i].getDiameter();
        for (size_t j = i + 1; j < N; j++)
        {
            double aj = particles[j].getDiameter();

            double dist = particles[i].distanceTo(particles[j]);
            double a = 0.5 * (ai + aj);

            if (dist > a)
            {
                f_prod *= 1 - a / dist;
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
double SphericalWF<N, d>::computeLaplacian(const ParticleRay<N, d> &particles)
{
    double sum = 0;
    for (size_t i = 0; i < N; i++)
    {
        double t1 = 4 * alpha_ * alpha_ * particles[i].getSquaredDistance();
        double t2 = -2 * alpha_ * (double)d;
        sum += t1 + t2;
    }

    return sum / evaluate(particles);
}

template <size_t N, size_t d>
double SphericalWF<N, d>::computeLocalEnergy(const ParticleRay<N, d> &particles)
{
    double hbar = 1;
    double m = 1;
    double omega = 1;

    double h2m = hbar * hbar / m;
    double mw2 = m * omega * omega;

    double sum = 0;
    for (auto &particle : particles)
    {
        double r2 = particle.getSquaredDistance();
        double t1 = h2m * ((double)d * alpha_ - 2 * alpha_ * alpha_ * r2);
        double t2 = 0.5 * mw2  * r2;
        sum += t1 + t2;
    }

    return sum;
}

template <size_t N, size_t d>
QForceMat<N, d> SphericalWF<N, d>::computeQForce(const ParticleRay<N, d> &particles)
{
    QForceMat<N, d> qForceMat;
    for (size_t i = 0; i < N; i++)
    {
        auto pos = particles[i].getPosition();
        for (size_t j = 0; j < d; j++)
        {
            qForceMat[i][j] = -4 * alpha_ * pos[j];
        }
    }

    return qForceMat;
}
