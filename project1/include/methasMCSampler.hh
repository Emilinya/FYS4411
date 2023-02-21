#pragma once

#include <array>
#include <tuple>
#include <math.h>
#include <omp.h>

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

// Monte Carlo sampling with the Metropolis-Hastings algorithm
template <size_t N, size_t d>
double methasMonteCarloStep(
    double timeStep, ParticleSystem<N, d> &stateOld, QForceMat<N, d> &qForceOld,
    double &wfOld, const WaveFunction<N, d> &waveFunction, Random &random)
{
    const double D = 0.5;  // Diffusion constant, =1/2 in atomic units 

    ParticleSystem<N, d> stateNew = stateOld;
    // Trial position moving one particle at the time
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < d; j++)
        {
            double rndStep = random.nextGaussian(0.0, 1.0) * sqrt(timeStep);
            double qForce = qForceOld[i][j] * timeStep * D;
            stateNew.adjustPostitionAt(i, rndStep + qForce, j);
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
            stateOld.setAt(i, stateNew[i]);
            qForceOld[i] = qForceNew[i];
            wfOld = wfNew;
        }
    }

    return waveFunction.computeLocalEnergy(stateOld);
}

template <size_t N, size_t d, class WFClass>
std::tuple<double, double> methasMonteCarloSampler(
    double timeStep, size_t mcCycleCount, size_t burnCycleCount,
    size_t walker_count, double diameter, double mass, double stateSize, double alpha)
{
    std::vector<double> energies;
    energies.resize(walker_count);

    # pragma omp parallel for
    for (size_t i = 0; i < walker_count; i++)
    {
        Random random;
        ParticleSystem<N, d> state(diameter, mass, stateSize);

        double energy = 0.0;
        WFClass waveFunction(alpha);
        double wf = waveFunction.evaluate(state);
        QForceMat<N, d> qForce = waveFunction.computeQForce(state);

        // burn some samples
        for (size_t i = 0; i < burnCycleCount; i++)
        {
            methasMonteCarloStep(timeStep, state, qForce, wf, waveFunction, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            double deltaE = methasMonteCarloStep(timeStep, state, qForce, wf, waveFunction, random);

            energy += deltaE;
        }

        energies[i] = energy / mcCycleCount;
    }

    return calcMeanStd(energies);
}

// template <size_t N, size_t d>
// void sphericalMCSampler(
//     double D, double timeStep, size_t mcCycleCount, std::vector<double> &alphaValues,
//     ParticleSystem<N, d> &initialState, SphericalWF<N, d> &waveFunction, Random &random)
// {
//     for (size_t ia = 0; ia < alphaValues.size(); ia++)
//     {
//         auto [energy, error] = monteCarloSampler(
//             D, timeStep, mcCycleCount, initialState, waveFunction, random);

//         // outfile.write('%f %f %f %f %f\n' % (alpha, beta, energy, variance, error));
//     }

//     // return energies, alphaValues, betaValues;
// }

// template <size_t N, size_t d>
// void elipticalMCSampler(
//     double D, double timeStep, size_t mcCycleCount, std::vector<double> &alphaValues,
//     std::vector<double> &betaValues, ParticleSystem<N, d> &initialState,
//     ElipticalWF<N, d> &waveFunction, Random &random)
// {
//     for (size_t ia = 0; ia < alphaValues.size(); ia++)
//     {
//         for (size_t ib = 0; ib < betaValues.size(); ib++)
//         {
//             auto [energy, error] = monteCarloSampler(
//                 D, timeStep, mcCycleCount, initialState, waveFunction, random);

//             // outfile.write('%f %f %f %f %f\n' % (alpha, beta, energy, variance, error));
//         }
//     }

//     // return energies, alphaValues, betaValues;
// }
