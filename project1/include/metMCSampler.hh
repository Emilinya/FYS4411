#pragma once

#include <array>
#include <tuple>
#include <math.h>
#include <omp.h>

#include "utils.hh"
#include "Random.hh"
#include "WaveFunction.hh"
#include "SphericalWF.hh"

// Monte Carlo sampling with the Metropolis algorithm
template <size_t N, size_t d>
double metMonteCarloStep(
    double stepSize, ParticleSystem<N, d> &stateOld, double &wfOld, const WaveFunction<N, d> &waveFunction, Random &random)
{
    ParticleSystem<N, d> stateNew = stateOld;
    // Trial position moving one particle at the time
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < d; j++)
        {
            double rndStep = stepSize * (random.nextDouble(0, 1) - 0.5);
            stateNew.adjustPostitionAt(i, rndStep, j);
        }

        double wfNew = waveFunction.evaluate(stateNew);

        double probabilityRatio = (wfNew * wfNew) / (wfOld * wfOld);

        // Metropolis-Hastings test to see whether we accept the move
        if (random.nextDouble(0, 1) <= probabilityRatio)
        {
            stateOld.setAt(i, stateNew[i]);
            wfOld = wfNew;
        }
    }

    return waveFunction.computeLocalEnergy(stateOld);
}

template <size_t N, size_t d, class WFClass>
std::tuple<double, double> metMonteCarloSampler(
    double stepSize, size_t mcCycleCount, size_t burnCycleCount,
    size_t walkerCount, double diameter, double mass, double stateSize, double alpha)
{
    std::vector<double> energies;
    energies.resize(walkerCount);

    #pragma omp parallel for
    for (size_t i = 0; i < walkerCount; i++)
    {
        Random random;
        ParticleSystem<N, d> state(diameter, mass, stateSize);

        double energy = 0.0;
        WFClass waveFunction(alpha);
        double wf = waveFunction.evaluate(state);

        // burn some samples
        for (size_t i = 0; i < burnCycleCount; i++)
        {
            metMonteCarloStep(stepSize, state, wf, waveFunction, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            double deltaE = metMonteCarloStep(stepSize, state, wf, waveFunction, random);

            energy += deltaE;
        }

        energies[i] = energy / mcCycleCount;
    }

    return calcMeanStd(energies);
}
