#pragma once

#include <array>
#include <tuple>
#include <math.h>

#include "utils.hh"
#include "Random.hh"
#include "WaveFunction.hh"
#include "SphericalWF.hh"

template <size_t d>
double calcGreensFunction(
    double D, double timeStep,
    const std::array<double, d> &qForceOld, const std::array<double, d> &qForceNew,
    const std::array<double, d> &posOld, const std::array<double, d> &posNew)
{
    double expTerm = 0.0;
    for (size_t j = 0; j < d; j++)
    {
        double factor = 0.5 * (qForceOld[j] + qForceNew[j]);
        double term1 = 0.5 * D * timeStep * (qForceOld[j] - qForceNew[j]);
        double term2 = posNew[j] + posOld[j];
        expTerm += factor * (term1 - term2);
    }

    return exp(expTerm);
}

template <size_t N, size_t d>
std::tuple<double, double> monteCarloSampler(
    double D, double timeStep, size_t mcCycleCount, ParticleRay<N, d> &initialState,
    WaveFunction<N, d> &waveFunction, Random &random)
{
    double energy = 0.0;
    double energy2 = 0.0;

    ParticleRay<N, d> stateOld = initialState;

    double wfOld = waveFunction.evaluate(stateOld);
    QForceMat<N, d> qForceOld = waveFunction.computeQForce(stateOld);

    // Loop over MC cycles
    for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
    {
        ParticleRay<N, d> stateNew = stateOld;
        // Trial position moving one particle at the time
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < d; j++)
            {
                double rndStep = random.nextGaussian(0.0, 1.0) * sqrt(timeStep);
                double qForce = qForceOld[i][j] * timeStep * D;
                stateNew[i].adjustPosition(rndStep + qForce, j);
            }

            double wfNew = waveFunction.evaluate(stateNew);
            QForceMat<N, d> qForceNew = waveFunction.computeQForce(stateNew);

            double greensFunction = calcGreensFunction<d>(
                D, timeStep, qForceOld[i], qForceNew[i],
                stateOld[i].getPosition(), stateNew[i].getPosition());
            double probabilityRatio = greensFunction * (wfNew * wfNew) / (wfOld * wfOld);

            // Metropolis-Hastings test to see whether we accept the move
            if (random.nextDouble(0, 1) <= probabilityRatio)
            {
                stateOld[i] = stateNew[i];
                qForceOld[i] = qForceNew[i];
                wfOld = wfNew;
            }
        }
        double deltaE = waveFunction.computeLocalEnergy(stateOld);
        energy += deltaE;
        energy2 += deltaE * deltaE;
    }

    energy /= mcCycleCount;
    energy2 /= mcCycleCount;

    return {energy, energy2};
}

template <size_t N, size_t d>
void sphericalMCSampler(
    double D, double timeStep, size_t mcCycleCount, std::vector<double> &alphaValues,
    ParticleRay<N, d> &initialState, SphericalWF<N, d> &waveFunction, Random &random)
{
    for (size_t ia = 0; ia < alphaValues.size(); ia++)
    {
        auto [energy, energy2] = monteCarloSampler(
            D, timeStep, mcCycleCount, initialState, waveFunction, random);

        double variance = energy2 - energy * energy;
        double error = sqrt(variance / (double)mcCycleCount);

        // outfile.write('%f %f %f %f %f\n' % (alpha, beta, energy, variance, error));
    }

    // return energies, alphaValues, betaValues;
}

template <size_t N, size_t d>
void elipticalMCSampler(
    double D, double timeStep, size_t mcCycleCount, std::vector<double> &alphaValues,
    std::vector<double> &betaValues, ParticleRay<N, d> &initialState,
    ElipticalWF<N, d> &waveFunction, Random &random)
{
    for (size_t ia = 0; ia < alphaValues.size(); ia++)
    {
        for (size_t ib = 0; ib < betaValues.size(); ib++)
        {
            auto [energy, energy2] = monteCarloSampler(
                D, timeStep, mcCycleCount, initialState, waveFunction, random);

            double variance = energy2 - energy * energy;
            double error = sqrt(variance / (double)mcCycleCount);

            // outfile.write('%f %f %f %f %f\n' % (alpha, beta, energy, variance, error));
        }
    }

    // return energies, alphaValues, betaValues;
}
