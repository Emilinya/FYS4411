#pragma once

#include <limits>
#include <assert.h>
#include <exception>

#include "Particle.hh"
#include "WaveFunction.hh"

template <size_t N>
class ElipticalWF : public WaveFunction<N, 3>
{
public:
    ElipticalWF(double alpha, double beta, double gamma, MCMode mode) : mode_(mode)
    {
        assert(alpha >= 0);

        alpha_ = alpha;
        beta_ = beta;
        gamma_ = gamma;

        Gamma_ = (gamma_ * gamma_ - 4. * (alpha_ * alpha_) * (beta_ * beta_)) / (1. - 4. * alpha_ * alpha_);

        this->parameters_ = {alpha, beta};
    }

    inline double betaSquaredLen(const Particle<3> &particle);

    void setState(const ParticleSystem<N, 3> &particleSystem);

    void pertubateState(size_t idx, double magnitude, Random &random);
    void updateFrom(WaveFunction<N, 3> &waveFunction, size_t idx);

    double evaluate();
    double computeLocalEnergy();
    QForceMat<N, 3> &computeQForce();
    std::vector<double> &computeLogGrad();

private:
    double alpha_ = 0;
    double beta_ = 0;
    double gamma_ = 0;
    double Gamma_ = 0;
    const MCMode mode_;

    std::array<std::array<Vec3, N>, N> upsilonTable;
};

template <size_t N>
inline double ElipticalWF<N>::betaSquaredLen(const Particle<3> &particle)
{
    auto &pos = particle.getPosition();
    return pos[0] * pos[0] + pos[1] * pos[1] + beta_ * pos[2] * pos[2];
}

template <size_t N>
void ElipticalWF<N>::setState(const ParticleSystem<N, 3> &particleSystem)
{
    this->state_ = particleSystem;

    this->value_.reset();
    this->qForce_.reset();
    this->logGrad_.reset();
    this->localEnergy_.reset();

    double a = particleSystem.getDiameter();

    // create a lookup table for upsilon values
    for (size_t i = 0; i < N; i++)
    {
        auto &pos1 = particleSystem[i].getPosition();
        for (size_t j = i + 1; j < N; j++)
        {
            auto &pos2 = particleSystem[j].getPosition();

            Vec3 diffVec;

            double diffLen = 0;
            for (size_t k = 0; k < 3; k++)
            {
                diffVec(k) = pos1[k] - pos2[k];
                diffLen += diffVec(k) * diffVec(k);
            }

            if (diffLen <= a * a)
            {
                // particles are colliding, oh no!
                throw std::runtime_error("ElipticalWF: Two particles are colliding, state is impossible!");
            }

            diffLen = std::sqrt(diffLen);
            double uPrime = a / ((diffLen - a) * diffLen);

            upsilonTable[i][j] = diffVec * uPrime / diffLen;
            upsilonTable[j][i] = -upsilonTable[i][j];
        }
    }
}

template <size_t N>
void ElipticalWF<N>::pertubateState(size_t idx, double magnitude, Random &random)
{
    if (!this->state_)
    {
        throw std::runtime_error("ElipticalWF: Can't pertubate state before setting state");
    }

    ParticleSystem<N, 3> &system = this->state_.value();
    double a = system.getDiameter();

    Particle<3> particleCopy = system[idx];
    for (size_t j = 0; j < 3; j++)
    {
        if (mode_ == MCMode::MET)
        {
            double rndStep = magnitude * (random.nextDouble(0, 1) - 0.5);
            particleCopy.adjustPosition(rndStep, j);
        }
        else if (mode_ == MCMode::METHAS)
        {
            double rndStep = random.nextGaussian(0.0, 1.0) * sqrt(magnitude);
            double qForce = computeQForce()[idx][j] * magnitude * 0.5; // 0.5: Diffusion constant in atomic units
            particleCopy.adjustPosition(rndStep + qForce, j);
        }
    }
    const auto &newPos = particleCopy.getPosition();

    // subtract old upsilon values from qForce
    if (mode_ == MCMode::METHAS)
    {
        QForceMat<N, 3> &qForce = computeQForce();

        for (size_t i = 0; i < N; i++)
        {
            if (i == idx)
            {
                continue;
            }

            Vec3 &upsilonVec = upsilonTable[i][idx];
            for (size_t k = 0; k < 3; k++)
            {
                qForce[i][k] -= 2. * upsilonVec(k);
            }
        }
    }

    // calculate new value and update upsilon table
    double prevValue = evaluate();
    double gChange = std::exp(-alpha_ * (betaSquaredLen(particleCopy) - betaSquaredLen(system[idx])));
    double fChange = 1;
    for (size_t i = 0; i < N; i++)
    {
        if (i == idx)
        {
            continue;
        }

        double diffLenOld = system[idx].distanceTo(system[i]);
        double diffLenNew = particleCopy.distanceTo(system[i]);

        if (diffLenNew <= a)
        {
            // particles are colliding, state is impossible!
            throw std::runtime_error("ElipticalWF: Two particles are colliding, state is impossible!");
        }

        fChange *= 1. - a * (1. - diffLenOld / diffLenNew) / (a - diffLenOld);

        Vec3 diffVecNew;
        for (size_t k = 0; k < 3; k++)
        {
            diffVecNew(k) = system[i].getPosition()[k] - newPos[k];
        }

        double uPrime = a / ((diffLenNew - a) * diffLenNew);

        upsilonTable[i][idx] = diffVecNew * uPrime / diffLenNew;
        upsilonTable[idx][i] = -upsilonTable[i][idx];
    }
    this->value_ = prevValue * gChange * fChange;

    // add new upsilon values to qForce and recalculate qForce[idx]
    if (mode_ == MCMode::METHAS)
    {
        QForceMat<N, 3> &qForce = this->getQForce().value();

        qForce[idx][0] = -4. * alpha_ * newPos[0];
        qForce[idx][1] = -4. * alpha_ * newPos[1];
        qForce[idx][2] = -4. * alpha_ * beta_ * newPos[2];

        for (size_t i = 0; i < N; i++)
        {
            if (i == idx)
            {
                continue;
            }

            Vec3 &upsilonVec = upsilonTable[i][idx];

            for (size_t k = 0; k < 3; k++)
            {
                qForce[i][k] += 2. * upsilonVec(k);
                qForce[idx][k] -= 2. * upsilonVec(k);
            }
        }
    }

    system.setAt(idx, particleCopy);
    this->localEnergy_.reset();
    this->logGrad_.reset();
}

template <size_t N>
void ElipticalWF<N>::updateFrom(WaveFunction<N, 3> &waveFunction, size_t idx)
{
    if (!this->state_)
    {
        throw std::runtime_error("ElipticalWF: Can't update from wavefunction before setting state");
    }

    ParticleSystem<N, 3> &thisState = this->state_.value();
    auto &newParticle = waveFunction.getState()[idx];
    auto &newPos = newParticle.getPosition();

    thisState.setAt(idx, newParticle);
    double a = thisState.getDiameter();

    this->value_ = waveFunction.getValue();
    this->qForce_ = waveFunction.getQForce();
    this->localEnergy_.reset();
    this->logGrad_.reset();

    // update upsilon table
    for (size_t i = 0; i < N; i++)
    {
        if (i == idx)
        {
            continue;
        }
        auto &oldPos = thisState[i].getPosition();

        Vec3 diffVec;

        double diffLen = 0;
        for (size_t k = 0; k < 3; k++)
        {
            diffVec(k) = oldPos[k] - newPos[k];
            diffLen += diffVec(k) * diffVec(k);
        }
        diffLen = std::sqrt(diffLen);

        double uPrime = a / ((diffLen - a) * diffLen);

        upsilonTable[i][idx] = diffVec * uPrime / diffLen;
        upsilonTable[idx][i] = -upsilonTable[i][idx];
    }
}

template <size_t N>
double ElipticalWF<N>::evaluate()
{
    if (this->value_)
    {
        return this->value_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("ElipticalWF: Can't evaluate before setting state");
    }

    ParticleSystem<N, 3> &system = this->state_.value();

    double betaSquareSum = 0;
    for (size_t i = 0; i < N; i++)
    {
        betaSquareSum += betaSquaredLen(system[i]);
    }
    double g_prod = std::exp(-alpha_ * betaSquareSum);

    double f_prod = 1;
    double a = system.getDiameter();
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = i + 1; j < N; j++)
        {
            f_prod *= 1 - a / system[i].distanceTo(system[j]);
        }
    }

    this->value_ = g_prod * f_prod;

    return this->value_.value();
}

template <size_t N>
double ElipticalWF<N>::computeLocalEnergy()
{
    if (this->localEnergy_)
    {
        return this->localEnergy_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("ElipticalWF: Can't compute local energy before setting state");
    }

    ParticleSystem<N, 3> &system = this->state_.value();

    // calculate local energy
    double term1 = 0;
    double term2 = 0;

    for (size_t i = 0; i < N; i++)
    {
        auto &posi = system[i].getPosition();
        term1 += posi[0] * posi[0] + posi[1] * posi[1] + Gamma_ * posi[2] * posi[2];

        for (size_t j = i + 1; j < N; j++)
        {
            auto &posj = system[j].getPosition();

            auto &vec1 = upsilonTable[i][j];

            Vec3 vec2;
            vec2(0) = 2. * alpha_ * (posi[0] - posj[0]);
            vec2(1) = 2. * alpha_ * (posi[1] - posj[1]);
            vec2(2) = 2. * alpha_ * beta_ * (posi[2] - posj[2]);

            Vec3 vsum = Vec3().fill(0.);
            for (size_t k = 0; k < N; k++)
            {
                if (k == i || k == j)
                {
                    continue;
                }
                vsum += upsilonTable[i][k] - upsilonTable[j][k];
            }

            vec2 -= 0.5 * vsum;

            term2 += arma::dot(vec1, vec2);
        }
    }

    return (double)N * alpha_ * (beta_ + 2.) + (0.5 - 2. * alpha_ * alpha_) * term1 + term2;
}

template <size_t N>
QForceMat<N, 3> &ElipticalWF<N>::computeQForce()
{
    if (this->qForce_)
    {
        return this->qForce_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("ElipticalWF: Can't compute QForce before setting state");
    }

    ParticleSystem<N, 3> &system = this->state_.value();

    QForceMat<N, 3> qForce;
    for (size_t i = 0; i < N; i++)
    {
        Vec3 upsilonSum = Vec3().fill(0.);
        for (size_t j = 0; j < N; j++)
        {
            if (j == i)
            {
                continue;
            }
            upsilonSum += upsilonTable[i][j];
        }

        auto pos = system[i].getPosition();
        qForce[i][0] = -4 * alpha_ * pos[0] + 2. * upsilonSum(0);
        qForce[i][1] = -4 * alpha_ * pos[1] + 2. * upsilonSum(1);
        qForce[i][2] = -4 * alpha_ * beta_ * pos[2] + 2. * upsilonSum(2);
    }

    this->qForce_ = qForce;

    return this->qForce_.value();
}

template <size_t N>
std::vector<double> &ElipticalWF<N>::computeLogGrad()
{
    if (this->logGrad_)
    {
        return this->logGrad_.value();
    }

    if (!this->state_)
    {
        throw std::runtime_error("ElipticalWF: Can't compute local energy before setting state");
    }

    ParticleSystem<N, 3> &system = this->state_.value();

    this->logGrad_ = {0, 0};
    for (size_t i = 0; i < N; i++)
    {
        auto &pos = system[i].getPosition();
        this->logGrad_.value()[0] -= betaSquaredLen(system[i]);
        this->logGrad_.value()[0] -= alpha_ * pos[2] * pos[2];
    }

    return this->logGrad_.value();
}
