#pragma once

#include <math.h>
#include <assert.h>
#include <exception>

#include "Particle.hh"
#include "WaveFunction.hh"

template <size_t N, size_t d, size_t M>
class RBMWF : public WaveFunction<N, d>
{
public:
    RBMWF(
        Vec<N*d> a, Vec<M> b, Mat<N*d, M> W, double sigma,
        MCMode mode, bool interactions = false) : mode_(mode)
    {
        a_ = a;
        b_ = b;
        W_ = W;
        sigma2_ = sigma * sigma;
        interactions_ = interactions;
    }

    bool setState(const ParticleSystem<N, d> &particleSystem);

    bool pertubateState(size_t idx, double magnitude, Random &random);
    void updateFrom(RBMWF<N, d, M> &waveFunction, size_t idx);

    double evaluate();
    double computeLocalEnergy();
    QForceMat<N, d> &computeQForce();

    RBMGrad<N, d, M> &computeLogGrad(const ParticleSystem<N, d> &particleSystem);
    inline std::optional<RBMGrad<N, d, M>> &getLogGrad() { return logGrad_; };
    RBMGrad<N, d, M> &computeLogGrad();


private:
    Vec<M> b_;
    Vec<N * d> a_;
    Mat<N * d, M> W_;
    double sigma2_;
    const MCMode mode_;
    bool interactions_;
    std::optional<RBMGrad<N, d, M>> logGrad_;

    Vec<N * d> x_;
    Vec<M> bWxExp_;
};

template <size_t N, size_t d, size_t M>
bool RBMWF<N, d, M>::setState(const ParticleSystem<N, d> &particleSystem)
{
    this->state_ = particleSystem;

    logGrad_.reset();
    this->value_.reset();
    this->qForce_.reset();
    this->localEnergy_.reset();

    for (size_t i = 0; i < N; i++)
    {
        auto &pos = particleSystem[i].getPosition();
        for (size_t j = 0; j < d; j++)
        {
            x_(i*d + j) = pos[j];
        }
    }

    bWxExp_ = arma::exp(b_ + (W_.t() * x_) / sigma2_);

    return true;
}

template <size_t N, size_t d, size_t M>
bool RBMWF<N, d, M>::pertubateState(size_t idx, double magnitude, Random &random)
{
    if (!this->state_)
    {
        throw std::runtime_error("RBMWF: Can't pertubate state before setting state");
    }

    ParticleSystem<N, d> &system = this->state_.value();

    // adjust position of particle at index idx
    Particle<d> particleCopy = system[idx];
    for (size_t j = 0; j < d; j++)
    {
        if (mode_ == MCMode::MET)
        {
            double rndStep = magnitude * (random.nextDouble(0, 1) - 0.5);
            particleCopy.adjustPosition(rndStep, j);
        }
        else if (mode_ == MCMode::METHAS)
        {
            double rndStep = random.nextGaussian(0.0, 1.0) * std::sqrt(magnitude);
            double qForce = computeQForce()[idx][j] * magnitude * 0.5; // 0.5: Diffusion constant in atomic units
            particleCopy.adjustPosition(rndStep + qForce, j);
        }
    }

    // see if any particles collide (this is very unlikely)
    for (size_t i = 0; i < N; i++)
    {
        if (i != idx && system[i].squareDistanceTo(particleCopy) == 0.0) {
            return false;
        }
    }

    // update lookup tables
    auto &oldPos = system[idx].getPosition();
    auto &newPos = particleCopy.getPosition();
    Vec<M> diffSum(arma::fill::zeros);
    for (size_t j = 0; j < d; j++)
    {
        diffSum += W_.row(j).t() * (oldPos[j] - newPos[j]);
        x_(idx * d + j) = newPos[j];
    }
    bWxExp_ %= arma::exp(-diffSum / sigma2_);

    // update system
    system.setAt(idx, particleCopy);

    // update value (because of the 1 + exp terms, there is no point in efficient computation)
    this->value_.reset();
    evaluate();

    // update quantum force (again, just reset and compute again)
    if (mode_ == MCMode::METHAS)
    {
        this->qForce_.reset();
        computeQForce();
    }

    // we only calculate the local energy and gradient once per MC cycle, so
    // we don't need to efficiently calculate them here
    this->localEnergy_.reset();
    logGrad_.reset();

    return true;
}

template <size_t N, size_t d, size_t M>
void RBMWF<N, d, M>::updateFrom(RBMWF<N, d, M> &waveFunction, size_t idx)
{
    if (!this->state_)
    {
        throw std::runtime_error("RBMWF: Can't update from wavefunction before setting state");
    }

    ParticleSystem<N, d> &thisState = this->state_.value();
    ParticleSystem<N, d> &otherState = waveFunction.getState();

    // update lookup tables
    auto &oldPos = thisState[idx].getPosition();
    auto &newPos = otherState[idx].getPosition();
    Vec<M> diffSum(arma::fill::zeros);
    for (size_t j = 0; j < d; j++)
    {
        diffSum += W_.row(j).t() * (oldPos[j] - newPos[j]);
        x_(idx * d + j) = newPos[j];
    }
    bWxExp_ %= arma::exp(-diffSum / sigma2_);

    // update system
    thisState.setAt(idx, otherState[idx]);

    // update value
    this->value_.reset();
    evaluate();

    // update quantum force
    if (mode_ == MCMode::METHAS)
    {
        this->qForce_.reset();
        computeQForce();
    }

    // reset local energy and gradients
    this->localEnergy_.reset();
    logGrad_.reset();
}

template <size_t N, size_t d, size_t M>
double RBMWF<N, d, M>::evaluate()
{
    if (this->value_)
    {
        return this->value_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("RBMWF: Can't evaluate before setting state");
    }

    auto diff = x_ - a_;
    double expTerm = std::exp(-arma::sum(diff % diff) / (2. * sigma2_));
    double prodTerm = arma::prod(1.0 + bWxExp_);

    this->value_ = expTerm * prodTerm;

    return this->value_.value();
}

template <size_t N, size_t d, size_t M>
double RBMWF<N, d, M>::computeLocalEnergy()
{
    if (this->localEnergy_)
    {
        return this->localEnergy_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("RBMWF: Can't compute local energy before setting state");
    }

    ParticleSystem<N, d> &system = this->state_.value();

    double omega2 = 1;

    Vec<N * d> diff = a_ - x_;
    Vec<M> expOexp = bWxExp_ / (1.0 + bWxExp_);

    double term1 = -0.5 / (sigma2_ * sigma2_) * arma::sum(diff % diff);
    double term2 = 0.5 * omega2 * arma::sum(x_ % x_);

    double term3 = 0;
    for (size_t i = 0; i < N*d; i++)
    {
        double iterm = 0;
        for (size_t l = 0; l < M; l++)
        {
            double lterm1 = 2.0 * diff(i);
            double lterm2 = W_(i, l) / (1.0 + bWxExp_(l));
            double lterm3 = 0;
            for (size_t m = 0; m < M; m++)
            {
                lterm3 += W_(i, m) * expOexp(m);
            }
            iterm += W_(i, l) * expOexp(l) * (lterm1 + lterm2 + lterm3);
        }
        term3 += iterm;
    }
    term3 *= -0.5 / (sigma2_ * sigma2_);

    double term4 = 0.5 * (double)(N * d) / sigma2_;

    double term5 = 0;
    if (interactions_)
    {
        for (size_t i = 0; i < N; i++)
        {
            auto &parti = system[i];
            for (size_t j = i+1; j < N; j++)
            {
                auto &partj = system[j];
                term5 += 1.0 / (parti.distanceTo(partj));
            }
        }
    }

    this->localEnergy_ = term1 + term2 + term3 + term4 + term5;
    

    return this->localEnergy_.value();
}

template <size_t N, size_t d, size_t M>
QForceMat<N, d> &RBMWF<N, d, M>::computeQForce()
{
    if (this->qForce_)
    {
        return this->qForce_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("RBMWF: Can't compute QForce before setting state");
    }

    Vec<M> expOexp = bWxExp_ / (1.0 + bWxExp_);

    QForceMat<N, d> qForceMat;
    for (size_t i = 0; i < N * d; i++)
    {
        size_t fi = i / d;
        size_t fj = i - d * fi;

        double diff = (a_(i) - x_(i)) / sigma2_;
        double sum = 0;
        for (size_t j = 0; j < M; j++)
        {
            sum += W_(i, j) / sigma2_ * expOexp(j);
        }

        qForceMat[fi][fj] = 2. * (diff + sum);
    }

    this->qForce_ = qForceMat;

    return this->qForce_.value();
}

template <size_t N, size_t d, size_t M>
RBMGrad<N, d, M> &RBMWF<N, d, M>::computeLogGrad()
{
    if (logGrad_)
    {
        return logGrad_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("RBMWF: Can't compute local energy before setting state");
    }

    RBMGrad<N, d, M> grad;
    grad.aGrad = (x_ - a_) / sigma2_;
    grad.bGrad = bWxExp_ / (1.0 + bWxExp_);
    grad.WGrad = (x_ / sigma2_) * grad.bGrad.t();
    
    logGrad_ = grad;

    return logGrad_.value();
}

template <size_t N, size_t d, size_t M>
RBMGrad<N, d, M> &RBMWF<N, d, M>::computeLogGrad(const ParticleSystem<N, d> &particleSystem)
{
    this->setState(particleSystem);
    return computeLogGrad();
}