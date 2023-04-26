#pragma once

#include <array>
#include <tuple>
#include <math.h>
#include <omp.h>

#include "utils.hh"
#include "RBMWF.hh"
#include "Random.hh"

template <size_t N, size_t d, size_t M>
double calcGreensFunction(
    double timeStep, size_t idx,
    RBMWF<N, d, M> &waveFunctionOld, RBMWF<N, d, M> &waveFunctionNew)
{
    const double D = 0.5; // Diffusion constant, 1/2 in atomic units

    QForceMat<N, d> &qForceOld = waveFunctionOld.computeQForce();
    QForceMat<N, d> &qForceNew = waveFunctionNew.computeQForce();

    const std::array<double, d> &posOld = waveFunctionOld.getState()[idx].getPosition();
    const std::array<double, d> &posNew = waveFunctionNew.getState()[idx].getPosition();

    double expTerm = 0.0;
    for (size_t j = 0; j < d; j++)
    {
        double factor = -0.25 * (qForceNew[idx][j] + qForceOld[idx][j]);
        double term1 = D * timeStep * (qForceNew[idx][j] - qForceOld[idx][j]);
        double term2 = 2. * (posNew[j] - posOld[j]);
        expTerm += factor * (term1 + term2);
    }

    return exp(expTerm);
}

template <size_t N, size_t d, size_t M>
MCStepOut<N, d, M> mcStep(
    double magnitude, RBMWF<N, d, M> &waveFunctionOld, const MCMode mode, Random &random)
{
    RBMWF<N, d, M> waveFunctionNew = waveFunctionOld;

    // trial position moving one particle at a time
    for (size_t i = 0; i < N; i++)
    {
        // perturbateState returns false if new state is impossible (particles are colliding)
        if (waveFunctionNew.pertubateState(i, magnitude, random))
        {
            double wfOld = waveFunctionOld.evaluate();
            double wfNew = waveFunctionNew.evaluate();

            double probabilityRatio = (wfNew * wfNew) / (wfOld * wfOld);
            if (mode == MCMode::METHAS)
            {
                probabilityRatio *= calcGreensFunction<N, d, M>(magnitude, i, waveFunctionOld, waveFunctionNew);
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
    }

    return {
        waveFunctionOld.computeLocalEnergy(),
        waveFunctionOld.computeLogGrad(),
    };
}

// Monte Carlo sampling of the spherical wave function using
// the Metropolis or Metropolis-Hastings algorithm
template <size_t N, size_t d, size_t M>
MCSamplerOut<N, d, M> mcSampler(
    const MCMode mode, double magnitude, size_t mcCycleCount, size_t burnCycleCount,
    size_t walkerCount, double sigma, bool interactions, RBMParams<N, d, M> &params,
    const bool saveSamples = false, std::string sampleFileName = "")
{
    std::vector<double> energies(walkerCount, 0);
    std::vector<RBMGrad<N, d, M>> gradMuls(walkerCount);
    std::vector<RBMGrad<N, d, M>> gradients(walkerCount);

    std::vector<std::vector<double>> samples;
    if (saveSamples)
    {
        samples = std::vector<std::vector<double>>(
            walkerCount, std::vector<double>(mcCycleCount, 0));
    }

#pragma omp parallel for
    for (size_t i = 0; i < walkerCount; i++)
    {
        Random random;
        ParticleSystem<N, d> state(sigma);

        double energy = 0.0;
        RBMGrad<N, d, M> gradMul;
        RBMGrad<N, d, M> gradient;

        RBMWF<N, d, M> waveFunction(params.a, params.b, params.W, sigma, mode, interactions);
        waveFunction.setState(state);

        // evaluate once so value is copied to the wave function copy
        waveFunction.evaluate();

        // burn some samples
        for (size_t mcCycle = 0; mcCycle < burnCycleCount; mcCycle++)
        {
            mcStep<N, d, M>(magnitude, waveFunction, mode, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            MCStepOut<N, d, M> out = mcStep<N, d, M>(magnitude, waveFunction, mode, random);
            if (saveSamples)
            {
                samples[i][mcCycle] = out.E;
            }

            energy += out.E;

            gradient.aGrad += out.logGrad.aGrad;
            gradient.bGrad += out.logGrad.bGrad;
            gradient.WGrad += out.logGrad.WGrad;

            gradMul.aGrad += out.logGrad.aGrad * out.E;
            gradMul.bGrad += out.logGrad.bGrad * out.E;
            gradMul.WGrad += out.logGrad.WGrad * out.E;
        }

        energies[i] = energy / mcCycleCount;

        gradients[i].aGrad = gradient.aGrad / mcCycleCount;
        gradients[i].bGrad = gradient.bGrad / mcCycleCount;
        gradients[i].WGrad = gradient.WGrad / mcCycleCount;

        gradMuls[i].aGrad = gradMul.aGrad / mcCycleCount;
        gradMuls[i].bGrad = gradMul.bGrad / mcCycleCount;
        gradMuls[i].WGrad = gradMul.WGrad / mcCycleCount;
    }

    std::vector<double> energyGrads(walkerCount, 0);

    RBMGrad<N, d, M> grad;
    for (size_t i = 0; i < walkerCount; i++)
    {
        grad.aGrad += 2 * (gradMuls[i].aGrad - gradients[i].aGrad * energies[i]);
        grad.bGrad += 2 * (gradMuls[i].bGrad - gradients[i].bGrad * energies[i]);
        grad.WGrad += 2 * (gradMuls[i].WGrad - gradients[i].WGrad * energies[i]);
    }
    grad.aGrad /= walkerCount;
    grad.bGrad /= walkerCount;
    grad.WGrad /= walkerCount;

    auto [E, Estd] = calcMeanStd(energies);

    if (saveSamples)
    {
        std::ofstream dataFile(sampleFileName);
        if (!dataFile)
        {
            std::cerr << "Error opening " + sampleFileName + ": " << strerror(errno) << std::endl;
            exit(0);
        }

        dataFile.precision(14);
        for (size_t j = 0; j < mcCycleCount; j++)
        {
            for (size_t i = 0; i < walkerCount; i++)
            {
                dataFile << samples[i][j] << " ";
            }
            dataFile << "\n";
        }
        dataFile.close();
    }

    return {E, Estd, grad};
}
