#pragma once

#include <memory>
#include <vector>
#include <optional>

#include "utils.hh"
#include "Particle.hh"
#include "ParticleSystem.hh"

template <size_t d>
class Particle;

template <size_t N, size_t d>
class WaveFunction
{
public:
    const std::vector<double> &getParameters() { return parameters_; }

    bool setState(const ParticleSystem<N, d> &particleSystem);
    inline ParticleSystem<N, d> &getState();

    virtual bool pertubateState(size_t idx, double magnitude, Random &random) = 0;
    virtual void updateFrom(WaveFunction<N, d> &waveFunction, size_t idx) = 0;

    double evaluate(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<double> &getValue() { return value_; };
    virtual double evaluate() = 0;

    double computeLocalEnergy(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<double> &getLocalEnergy() { return localEnergy_; };
    virtual double computeLocalEnergy() = 0;

    QForceMat<N, d> &computeQForce(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<QForceMat<N, d>> &getQForce() { return qForce_; };
    virtual QForceMat<N, d> &computeQForce() = 0;

    std::vector<double> &computeLogGrad(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<std::vector<double>> &getLogGrad() { return logGrad_; };
    virtual std::vector<double> &computeLogGrad() = 0;

protected:
    std::vector<double> parameters_;

    std::optional<double> value_;
    std::optional<double> localEnergy_;
    std::optional<QForceMat<N, d>> qForce_;
    std::optional<ParticleSystem<N, d>> state_;
    std::optional<std::vector<double>> logGrad_;
};

template <size_t N, size_t d>
inline bool WaveFunction<N, d>::setState(const ParticleSystem<N, d> &particleSystem)
{
    state_ = particleSystem;

    value_.reset();
    qForce_.reset();
    logGrad_.reset();
    localEnergy_.reset();

    return true;
}

template <size_t N, size_t d>
inline ParticleSystem<N, d> &WaveFunction<N, d>::getState()
{
    if (!state_)
    {
        throw std::runtime_error("WaveFunction: Can't get state before setting state");
    }
    return state_.value();
}

template <size_t N, size_t d>
double WaveFunction<N, d>::evaluate(const ParticleSystem<N, d> &particleSystem)
{
    setState(particleSystem);
    return evaluate();
}

template <size_t N, size_t d>
double WaveFunction<N, d>::computeLocalEnergy(const ParticleSystem<N, d> &particleSystem)
{
    setState(particleSystem);
    return computeLocalEnergy();
}

template <size_t N, size_t d>
QForceMat<N, d> &WaveFunction<N, d>::computeQForce(const ParticleSystem<N, d> &particleSystem)
{
    setState(particleSystem);
    return computeQForce();
}

template <size_t N, size_t d>
std::vector<double> &WaveFunction<N, d>::computeLogGrad(const ParticleSystem<N, d> &particleSystem)
{
    setState(particleSystem);
    return computeLogGrad();
}
