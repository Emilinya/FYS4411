#pragma once

#include <memory>
#include <vector>
#include <array>

#include "utils.hh"
#include "particle.hh"

template <size_t d>
class Particle;

template <size_t N, size_t d>
class WaveFunction
{
public:
    const std::vector<double> &getParameters() { return parameters_; }
    virtual double evaluate(const ParticleRay<N, d> &particles) = 0;
    virtual double computeLaplacian(const ParticleRay<N, d> &particles) = 0;

protected:
    std::vector<double> parameters_;
};

template <size_t N, size_t d>
class SphericalWF : public WaveFunction<N, d>
{
public:
    SphericalWF(double alpha);
    double evaluate(const ParticleRay<N, d> &particles);
    double computeLaplacian(const ParticleRay<N, d> &particles);

private:
    double alpha_ = 0;
};

template <size_t N, size_t d>
class ElipticalWF : public WaveFunction<N, d>
{
public:
    ElipticalWF(double alpha, double beta);
    double evaluate(const ParticleRay<N, d> &particles);
    double computeLaplacian(const ParticleRay<N, d> &particles);

private:
    double alpha_ = 0;
    double beta_ = 0;
};
