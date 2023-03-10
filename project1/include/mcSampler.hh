#pragma once

#include <array>
#include <tuple>
#include <math.h>
#include <omp.h>

#include "utils.hh"
#include "Random.hh"
#include "WaveFunction.hh"
#include "SphericalWF.hh"
#include "ElipticalWF.hh"

template <size_t N, size_t d>
double calcGreensFunction(
    double timeStep, size_t idx,
    WaveFunction<N, d> &waveFunctionOld, WaveFunction<N, d> &waveFunctionNew)
{
    const double D = 0.5; // Diffusion constant, 1/2 in atomic units

    QForceMat<N, d> &qForceOld = waveFunctionOld.computeQForce();
    QForceMat<N, d> &qForceNew = waveFunctionNew.computeQForce();

    const std::array<double, d> &posOld = waveFunctionOld.getState()[idx].getPosition();
    const std::array<double, d> &posNew = waveFunctionNew.getState()[idx].getPosition();

    double expTerm = 0.0;
    for (size_t j = 0; j < d; j++)
    {
        double factor = 0.5 * (qForceOld[idx][j] + qForceNew[idx][j]);
        double term1 = 0.5 * D * timeStep * (qForceOld[idx][j] - qForceNew[idx][j]);
        double term2 = posNew[j] + posOld[j];
        expTerm += factor * (term1 - term2);
    }

    return exp(expTerm);
}

template <size_t N, size_t d, class WFClass>
MCStepOut mcStep(
    double magnitude, WFClass &waveFunctionOld, const MCMode mode, Random &random)
{
    WFClass waveFunctionNew = waveFunctionOld;

    // trial position moving one particle at the time
    for (size_t i = 0; i < N; i++)
    {
        waveFunctionNew.pertubateState(i, magnitude, random);

        double wfOld = waveFunctionOld.evaluate();
        double wfNew = waveFunctionNew.evaluate();

        double probabilityRatio = (wfNew * wfNew) / (wfOld * wfOld);
        if (mode == MCMode::METHAS)
        {
            probabilityRatio *= calcGreensFunction<N, d>(magnitude, i, waveFunctionOld, waveFunctionNew);
        }

        // Metropolis-Hastings test to see whether we accept the move
        if (probabilityRatio >= 1 || random.nextDouble(0, 1) <= probabilityRatio)
        {
            waveFunctionOld.updateFrom(waveFunctionNew, i);
        }
        else
        {
            waveFunctionNew.updateFrom(waveFunctionOld, i);
        }
    }

    return {
        waveFunctionOld.computeLocalEnergy(),
        waveFunctionOld.computeLogGrad(),
    };
}

// Monte Carlo sampling with the Metropolis or Metropolis-Hastings algorithm
template <size_t N, size_t d, class WFClass>
MCSamplerOut mcSampler(
    const MCMode mode, double magnitude, size_t mcCycleCount, size_t burnCycleCount,
    size_t walkerCount, double diameter, double mass, double stateSize, double alpha)
{
    uint numParams = 0;
    if (typeid(WFClass) == typeid(SphericalWF<N, d>))
    {
        numParams = 1;
    }
    else if (typeid(WFClass) == typeid(ElipticalWF<N>))
    {
        numParams = 2;
    }

    std::vector<double> energies(walkerCount, 0);
    std::vector<std::vector<double>> gradMuls(walkerCount, std::vector<double>(numParams, 0));
    std::vector<std::vector<double>> gradients(walkerCount, std::vector<double>(numParams, 0));

#pragma omp parallel for
    for (size_t i = 0; i < walkerCount; i++)
    {
        Random random;
        ParticleSystem<N, d> state(diameter, mass, stateSize);

        double energy = 0.0;
        std::vector<double> gradMul(numParams, 0);
        std::vector<double> gradient(numParams, 0);

        WFClass waveFunction(alpha, mode);
        waveFunction.setState(state);

        // evaluate once so value is copied to the wave function copy
        waveFunction.evaluate();

        // burn some samples
        for (size_t mcCycle = 0; mcCycle < burnCycleCount; mcCycle++)
        {
            mcStep<N, d, WFClass>(magnitude, waveFunction, mode, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            MCStepOut out = mcStep<N, d, WFClass>(magnitude, waveFunction, mode, random);

            energy += out.E;
            for (size_t j = 0; j < numParams; j++)
            {
                gradient[j] += out.logGrad[j];
                gradMul[j] += out.E * out.logGrad[j];
            }
        }

        energies[i] = energy / mcCycleCount;

        for (size_t j = 0; j < numParams; j++)
        {
            gradMuls[i][j] = gradMul[j] / mcCycleCount;
            gradients[i][j] = gradient[j] / mcCycleCount;
        }
    }

    std::vector<std::vector<double>> energyGrads(walkerCount, std::vector<double>(numParams, 0));

    for (size_t i = 0; i < walkerCount; i++)
    {
        for (size_t j = 0; j < numParams; j++)
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
