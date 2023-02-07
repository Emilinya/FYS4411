#include <math.h>
#include <assert.h>
#include "include/particle.hh"
#include "include/waveFunction.hh"

template <size_t N, size_t d>
SphericalWF<N, d>::SphericalWF(double alpha)
{
    assert(alpha >= 0);

    alpha_ = alpha;
    this->parameters_ = {alpha};
}

template <size_t N, size_t d>
double SphericalWF<N, d>::evaluate(const ParticleRay<N, d> &particles)
{
    if (N == 1)
    {
        return exp(-alpha_ * particles[0]->getSquaredDistance());
    }

    double g_prod;
    double f_prod;

    for (size_t i = 0; i < N; i++)
    {
        g_prod *= exp(-alpha_ * particles[i]->getSquaredDistance());

        double ai = particles[i]->getDiameter();
        for (size_t j = i + 1; j < N; j++)
        {
            double aj = particles[j]->getDiameter();

            double dist = particles[i]->distanceTo(particles[j]);
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
    // we have only calculated the laplacian for a = 0
    for (auto &particle : particles)
    {
        assert(particle->getDiameter() == 0);
    }

    double sum = 0;
    for (size_t i = 0; i < N; i++)
    {
        double t1 = 4 * alpha_ * alpha_ * particles[i]->getSquaredDistance();
        double t2 = -2 * alpha_ * (double)d;
        sum += t1 + t2;
    }

    return sum / evaluate(particles);
}
