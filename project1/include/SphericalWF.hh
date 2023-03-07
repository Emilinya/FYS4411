#pragma once

#include <math.h>
#include <assert.h>
#include <exception>

#include "Particle.hh"
#include "WaveFunction.hh"

template <size_t N, size_t d>
class SphericalWF : public WaveFunction<N, d>
{
public:
    SphericalWF(double alpha, WFMode mode) : mode_(mode)
    {
        assert(alpha >= 0);

        alpha_ = alpha;
        this->parameters_ = {alpha};
    }

    void pertubateState(size_t idx, double magnitude, Random &random);
    void updateFrom(WaveFunction<N, d> &waveFunction, size_t idx);

    double evaluate();
    double computeLocalEnergy();
    QForceMat<N, d> &computeQForce();
    std::vector<double> &computeLogGrad();

private:
    double alpha_ = 0;
    const WFMode mode_;
};

template <size_t N, size_t d>
void SphericalWF<N, d>::pertubateState(size_t idx, double magnitude, Random &random)
{
    if (!this->state_)
    {
        throw std::runtime_error("SphericalWF: Can't pertubate state before setting state");
    }

    ParticleSystem<N, d> &system = this->state_.value();

    Particle<d> particleCopy = system[idx];
    for (size_t j = 0; j < d; j++)
    {
        if (mode_ == WFMode::MET)
        {
            double rndStep = magnitude * (random.nextDouble(0, 1) - 0.5);
            particleCopy.adjustPosition(rndStep, j);
        }
        else if (mode_ == WFMode::METHAS)
        {
            double rndStep = random.nextGaussian(0.0, 1.0) * sqrt(magnitude);
            double qForce = computeQForce()[idx][j] * magnitude * 0.5; // 0.5: Diffusion constant in atomic units
            particleCopy.adjustPosition(rndStep + qForce, j);
        }
    }

    double prevValue = evaluate();
    double distDiff = particleCopy.getSquaredDistance() - system[idx].getSquaredDistance();
    this->value_ = prevValue * std::exp(-alpha_ * distDiff);

    if (mode_ == WFMode::METHAS)
    {
        QForceMat<N, d> &qForce = computeQForce();
        const std::array<double, d> &newPos = particleCopy.getPosition();

        for (size_t j = 0; j < d; j++)
        {
            qForce[idx][j] = -4 * alpha_ * newPos[j];
        }
    }

    system.setAt(idx, particleCopy);
    this->localEnergy_.reset();
    this->logGrad_.reset();
}

template <size_t N, size_t d>
void SphericalWF<N, d>::updateFrom(WaveFunction<N, d> &waveFunction, size_t idx) {
    if (!this->state_)
    {
        throw std::runtime_error("SphericalWF: Can't update from wavefunction before setting state");
    }

    ParticleSystem<N, d> &thisState = this->state_.value();
    ParticleSystem<N, d> &otherState = waveFunction.getState();

    thisState.setAt(idx, otherState[idx]);

    this->value_ = waveFunction.getValue();
    this->localEnergy_.reset();
    this->logGrad_.reset();

    if (this->mode_ == WFMode::METHAS) {
        std::optional<QForceMat<N, d>> &otherQForce = waveFunction.getQForce();
        if (otherQForce) {
            if (this->qForce_) {
                this->qForce_.value()[idx] = otherQForce.value()[idx];
            } else {
                this->qForce_ = otherQForce;
            }
        } else {
            this->qForce_.reset();
        }
    }
}

template <size_t N, size_t d>
double SphericalWF<N, d>::evaluate()
{
    if (this->value_)
    {
        return this->value_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("SphericalWF: Can't evaluate before setting state");
    }

    ParticleSystem<N, d> &system = this->state_.value();

    double g_prod = std::exp(-alpha_ * system.getSquareSum());
    double a = system.getDiameter();

    if (a == 0)
    {
        // a = 0, we don't need to worry about interactions
        this->value_ = g_prod;
        this->value_.value();
    }

    double f_prod = 1;

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = i + 1; j < N; j++)
        {
            double dist2 = system[i].squareDistanceTo(system[j]);

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

    this->value_ = g_prod * f_prod;

    return this->value_.value();
}

template <size_t N, size_t d>
double SphericalWF<N, d>::computeLocalEnergy()
{
    if (this->localEnergy_)
    {
        return this->localEnergy_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("SphericalWF: Can't compute local energy before setting state");
    }

    ParticleSystem<N, d> &system = this->state_.value();

    double m = 1;
    double omega = 1;

    double const1 = alpha_ * (double)(d * N) / m;
    double const2 = 0.5 * m * omega * omega - 2. * alpha_ * alpha_ / m;

    double distSum = system.getSquareSum();

    this->localEnergy_ = const1 + const2 * distSum;

    return this->localEnergy_.value();
}

template <size_t N, size_t d>
QForceMat<N, d> &SphericalWF<N, d>::computeQForce()
{
    if (this->qForce_)
    {
        return this->qForce_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("SphericalWF: Can't compute QForce before setting state");
    }

    ParticleSystem<N, d> &system = this->state_.value();

    QForceMat<N, d> qForceMat;
    for (size_t i = 0; i < N; i++)
    {
        auto pos = system[i].getPosition();
        for (size_t j = 0; j < d; j++)
        {
            qForceMat[i][j] = -4 * alpha_ * pos[j];
        }
    }

    this->qForce_ = qForceMat;

    return this->qForce_.value();
}

template <size_t N, size_t d>
std::vector<double> &SphericalWF<N, d>::computeLogGrad()
{
    if (this->logGrad_)
    {
        return this->logGrad_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("SphericalWF: Can't compute local energy before setting state");
    }

    ParticleSystem<N, d> &system = this->state_.value();

    this->logGrad_ = {-system.getSquareSum()};
    return this->logGrad_.value();
}
