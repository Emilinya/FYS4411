#pragma once

#include <memory>
#include <vector>
#include <array>
#include <optional>

#include "utils.hh"
#include "Particle.hh"
#include "ParticleSystem.hh"

template <size_t d>
class Particle;

enum class WFMode {
    MET,
    METHAS,
};

template <size_t N, size_t d>
class WaveFunction
{
public:
    const std::vector<double> &getParameters() { return parameters_; }
    
    virtual void setState(const ParticleSystem<N, d> &particles) = 0;
    virtual inline ParticleSystem<N, d> &getState() = 0;

    virtual void pertubateState(size_t idx, double magnitude, Random &random) = 0;
    virtual void updateFrom(WaveFunction<N, d> &waveFunction, size_t idx) = 0;
    
    virtual double evaluate(const ParticleSystem<N, d> &particles) = 0;
    virtual double evaluate() = 0;
    virtual inline std::optional<double> &getValue() = 0;

    virtual double computeLocalEnergy(const ParticleSystem<N, d> &particles) = 0;
    virtual double computeLocalEnergy() = 0;
    virtual inline std::optional<double> &getLocalEnergy() = 0;

    virtual QForceMat<N, d> &computeQForce(const ParticleSystem<N, d> &particles) = 0;
    virtual QForceMat<N, d> &computeQForce() = 0;
    virtual inline std::optional<QForceMat<N, d>> &getQForce() = 0;

protected:
    std::vector<double> parameters_;

    std::optional<double> value_;
    std::optional<double> localEnergy_;
    std::optional<QForceMat<N, d>> qForce_;
    std::optional<ParticleSystem<N, d>> state_;
};

template <size_t N, size_t d>
class ElipticalWF : public WaveFunction<N, d>
{
public:
    ElipticalWF(double alpha, double beta, WFMode mode);

    void setState(const ParticleSystem<N, d> &particles);
    inline ParticleSystem<N, d> &getState();

    void pertubateState(size_t idx, double timeStep, Random &random);
    void updateFrom(WaveFunction<N, d> &waveFunction, size_t idx);

    double evaluate(const ParticleSystem<N, d> &particles);
    double evaluate();
    inline std::optional<double> &getValue();

    double computeLocalEnergy(const ParticleSystem<N, d> &particles);
    double computeLocalEnergy();
    inline std::optional<double> &getLocalEnergy();

    QForceMat<N, d> &computeQForce(const ParticleSystem<N, d> &particles);
    QForceMat<N, d> &computeQForce();
    inline std::optional<QForceMat<N, d>> &getQForce();

private:
    double alpha_ = 0;
    double beta_ = 0;
    const WFMode mode_;
};
