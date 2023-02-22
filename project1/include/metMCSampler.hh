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
template <size_t N, class WFClass>
double metMonteCarloStep(
    double stepSize, WFClass &waveFunctionOld, Random &random)
{
    WFClass waveFunctionNew = waveFunctionOld;
    double wfOld = waveFunctionOld.evaluate();

    for (size_t i = 0; i < N; i++)
    {
        waveFunctionNew.pertubateState(i, stepSize, random);

        double wfNew = waveFunctionNew.evaluate();
        double probabilityRatio = (wfNew * wfNew) / (wfOld * wfOld);

        // Metropolis-Hastings test to see whether we accept the move
        if (probabilityRatio >= 1 || random.nextDouble(0, 1) <= probabilityRatio)
        {
            waveFunctionOld.updateFrom(waveFunctionNew, i);
            wfOld = wfNew;
        }
    }

    return waveFunctionOld.computeLocalEnergy();
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
        WFClass waveFunction(alpha, WFMode::MET);
        waveFunction.setState(state);

        // burn some samples
        for (size_t i = 0; i < burnCycleCount; i++)
        {
            metMonteCarloStep<N, WFClass>(stepSize, waveFunction, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            double deltaE = metMonteCarloStep<N, WFClass>(stepSize, waveFunction, random);

            energy += deltaE;
        }

        energies[i] = energy / mcCycleCount;
    }

    return calcMeanStd(energies);
}
