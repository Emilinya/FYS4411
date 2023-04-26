#pragma once

#include <memory>
#include <vector>
#include <optional>

#include "utils.hh"
#include "Particle.hh"
#include "ParticleSystem.hh"

// The wavef unction class is more complicated than one might expect. Instead of just
// having functions that take in a particle system, the wavef unction class has ownership over
// an internal state. This is to enable efficient perturbation, when calculating a value from
// a particle system, the wavef unction copies the system and stores the computed value. You can
// then perturbate the stored particle system (called the state) using the pertubateState
// function, which will then efficiently calculate new values based on the stored values 
template <size_t N, size_t d>
class WaveFunction
{
public:
    const std::vector<double> &getParameters() { return parameters_; }

    // setState returns a bool, because the given state might be impossible. In that case,
    // setState returns false
    bool setState(const ParticleSystem<N, d> &particleSystem);
    inline ParticleSystem<N, d> &getState();

    // pertubateState returns a bool, because in might perturbate the state into an impossible state.
    // In that case, pertubateState returns false
    virtual bool pertubateState(size_t idx, double magnitude, Random &random) = 0;

    double evaluate(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<double> &getValue() { return value_; };
    virtual double evaluate() = 0;

    double computeLocalEnergy(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<double> &getLocalEnergy() { return localEnergy_; };
    virtual double computeLocalEnergy() = 0;

    QForceMat<N, d> &computeQForce(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<QForceMat<N, d>> &getQForce() { return qForce_; };
    virtual QForceMat<N, d> &computeQForce() = 0;

protected:
    std::vector<double> parameters_;

    std::optional<double> value_;
    std::optional<double> localEnergy_;
    std::optional<QForceMat<N, d>> qForce_;
    std::optional<ParticleSystem<N, d>> state_;
};

template <size_t N, size_t d>
inline bool WaveFunction<N, d>::setState(const ParticleSystem<N, d> &particleSystem)
{
    state_ = particleSystem;

    value_.reset();
    qForce_.reset();
    localEnergy_.reset();

    // this default implementation assumes that there are no impossible states
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
