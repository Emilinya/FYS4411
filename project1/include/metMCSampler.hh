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
MCStepOut metMonteCarloStep(
    double stepSize, WFClass &waveFunctionOld, Random &random)
{
    WFClass waveFunctionNew = waveFunctionOld;

    for (size_t i = 0; i < N; i++)
    {
        waveFunctionNew.pertubateState(i, stepSize, random);

        double wfOld = waveFunctionOld.evaluate();
        double wfNew = waveFunctionNew.evaluate();
        double probabilityRatio = (wfNew * wfNew) / (wfOld * wfOld);

        // Metropolis-Hastings test to see whether we accept the move
        if (probabilityRatio >= 1 || random.nextDouble(0, 1) <= probabilityRatio)
        {
            waveFunctionOld.updateFrom(waveFunctionNew, i);
        } else {
            waveFunctionNew.updateFrom(waveFunctionOld, i);
        }
    }

    return {
        waveFunctionOld.computeLocalEnergy(),
        waveFunctionOld.computeLogGrad(),
    };
}

template <size_t N, size_t d, class WFClass>
MCSamplerOut metMonteCarloSampler(
    double stepSize, size_t mcCycleCount, size_t burnCycleCount,
    size_t walkerCount, double diameter, double mass, double stateSize, double alpha)
{
    std::vector<double> energies;
    std::vector<std::vector<double>> gradMuls;
    std::vector<std::vector<double>> gradients;

    energies.resize(walkerCount);
    gradMuls.resize(walkerCount);
    gradients.resize(walkerCount);

    for (size_t i = 0; i < walkerCount; i++)
    {
        if (typeid(WFClass) == typeid(SphericalWF<N, d>))
        {
            gradients[i] = {0};
            gradMuls[i] = {0};
        }
        else if (typeid(WFClass) == typeid(ElipticalWF<N, d>))
        {
            gradients[i] = {0, 0};
            gradMuls[i] = {0, 0};
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < walkerCount; i++)
    {
        Random random;
        ParticleSystem<N, d> state(diameter, mass, stateSize);

        double energy = 0.0;
        std::vector<double> gradMul;
        std::vector<double> gradient;

        if (typeid(WFClass) == typeid(SphericalWF<N, d>))
        {
            gradMul = {0};
            gradient = {0};
        }
        else if (typeid(WFClass) == typeid(ElipticalWF<N, d>))
        {
            gradMul = {0, 0};
            gradient = {0, 0};
        }

        WFClass waveFunction(alpha, WFMode::MET);
        waveFunction.setState(state);

        // evaluate once so value is copied to the wave function copy
        waveFunction.evaluate();

        // burn some samples
        for (size_t mcCycle = 0; mcCycle < burnCycleCount; mcCycle++)
        {
            metMonteCarloStep<N, WFClass>(stepSize, waveFunction, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            MCStepOut out = metMonteCarloStep<N, WFClass>(stepSize, waveFunction, random);

            energy += out.E;
            for (size_t j = 0; j < out.logGrad.size(); j++)
            {
                gradient[j] += out.logGrad[j];
                gradMul[j] += out.E * out.logGrad[j];
            }
        }

        energies[i] = energy / mcCycleCount;

        for (size_t j = 0; j < gradient.size(); j++)
        {
            gradMuls[i][j] = gradMul[j] / mcCycleCount;
            gradients[i][j] = gradient[j] / mcCycleCount;
        }
    }

    std::vector<std::vector<double>> energyGrads;
    energyGrads.resize(walkerCount);

    for (size_t i = 0; i < walkerCount; i++)
    {
        energyGrads[i].resize(gradients[0].size());
        for (size_t j = 0; j < gradients[0].size(); j++)
        {
            energyGrads[i][j] = 2 * (gradMuls[i][j] - gradients[i][j] * energies[i]);
        }
    }

    auto energyTouple = calcMeanStd(energies);
    auto gradTouple = vectorCalcMeanStd(energyGrads);

    return {
        std::get<0>(energyTouple),
        std::get<1>(energyTouple),
        std::get<0>(gradTouple),
        std::get<1>(gradTouple),
    };
}
