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
        double factor = -0.25 * (qForceNew[idx][j] + qForceOld[idx][j]);
        double term1 = D * timeStep * (qForceNew[idx][j] - qForceOld[idx][j]);
        double term2 = 2. * (posNew[j] - posOld[j]);
        expTerm += factor * (term1 + term2);
    }

    return exp(expTerm);
}

template <size_t N, size_t d, class WFClass>
MCStepOut mcStep(
    double magnitude, WFClass &waveFunctionOld, const MCMode mode, Random &random)
{
    WFClass waveFunctionNew = waveFunctionOld;

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
    }

    return {
        waveFunctionOld.computeLocalEnergy(),
        waveFunctionOld.computeLogGrad(),
    };
}

// Monte Carlo sampling with the Metropolis or Metropolis-Hastings algorithm
template <size_t N, size_t d>
MCSamplerOut mcSampler(
    const MCMode mode, double magnitude, size_t mcCycleCount, size_t burnCycleCount,
    size_t walkerCount, double diameter, double stateSize, double alpha,
    const bool saveSamples = false, std::string sampleFileName = "")
{
    std::vector<double> energies(walkerCount, 0);
    std::vector<double> gradMuls(walkerCount, 0);
    std::vector<double> gradients(walkerCount, 0);

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
        ParticleSystem<N, d> state(diameter, stateSize);

        double energy = 0.0;
        double gradMul = 0.0;
        double gradient = 0.0;

        SphericalWF<N, d> waveFunction(alpha, mode);
        waveFunction.setState(state);

        // evaluate once so value is copied to the wave function copy
        waveFunction.evaluate();

        // burn some samples
        for (size_t mcCycle = 0; mcCycle < burnCycleCount; mcCycle++)
        {
            mcStep<N, d, SphericalWF<N, d>>(magnitude, waveFunction, mode, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            MCStepOut out = mcStep<N, d, SphericalWF<N, d>>(magnitude, waveFunction, mode, random);
            if (saveSamples)
            {
                samples[i][mcCycle] = out.E;
            }

            energy += out.E;
            gradient += out.logGrad[0];
            gradMul += out.E * out.logGrad[0];
        }

        energies[i] = energy / mcCycleCount;
        gradMuls[i] = gradMul / mcCycleCount;
        gradients[i] = gradient / mcCycleCount;
    }

    std::vector<double> energyGrads(walkerCount, 0);

    for (size_t i = 0; i < walkerCount; i++)
    {
        energyGrads[i] = 2 * (gradMuls[i] - gradients[i] * energies[i]);
    }

    /*
    both calcMeanStd and vectorCalcMeanStd calculate the standard deviation
    of the samples, not the standard deviation of the means. This means that
    the uncertanties are sqrt(walkerCount) too large. I did not notice this
    until I had ran all the calculations, so instead of fixing it here,
    I fixed it in the plotting program.
    */
    auto energyTouple = calcMeanStd(energies);
    auto gradTouple = calcMeanStd(energyGrads);

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

    return {
        std::get<0>(energyTouple),
        std::get<1>(energyTouple),
        {std::get<0>(gradTouple)},
        {std::get<1>(gradTouple)},
    };
}

// Monte Carlo sampling, now with ElipticalWF
template <size_t N>
MCSamplerOut mcSampler(
    const MCMode mode, double magnitude, size_t mcCycleCount, size_t burnCycleCount,
    size_t walkerCount, double diameter, double stateSize, double alpha, double beta,
    double gamma, const bool saveSamples = false, std::string sampleFileName = "")
{
    std::vector<double> energies(walkerCount, 0);
    std::vector<std::vector<double>> gradMuls(walkerCount, std::vector<double>(2, 0));
    std::vector<std::vector<double>> gradients(walkerCount, std::vector<double>(2, 0));

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

        double energy = 0.0;
        std::vector<double> gradMul(2, 0);
        std::vector<double> gradient(2, 0);

        ElipticalWF<N> waveFunction(alpha, beta, gamma, mode);
        while (true)
        {
            // generated state might be impossible. In that case, try again.
            ParticleSystem<N, 3> state(diameter, stateSize);
            if (waveFunction.setState(state))
            {
                break;
            };
        }

        // evaluate once so value is copied to the wave function copy
        waveFunction.evaluate();

        // burn some samples
        for (size_t mcCycle = 0; mcCycle < burnCycleCount; mcCycle++)
        {
            mcStep<N, 3, ElipticalWF<N>>(magnitude, waveFunction, mode, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            MCStepOut out = mcStep<N, 3, ElipticalWF<N>>(magnitude, waveFunction, mode, random);
            if (saveSamples)
            {
                samples[i][mcCycle] = out.E;
            }

            energy += out.E;

            gradient[0] += out.logGrad[0];
            gradient[1] += out.logGrad[1];
            gradMul[0] += out.E * out.logGrad[0];
            gradMul[1] += out.E * out.logGrad[1];
        }

        energies[i] = energy / mcCycleCount;

        gradMuls[i][0] = gradMul[0] / mcCycleCount;
        gradMuls[i][1] = gradMul[1] / mcCycleCount;
        gradients[i][0] = gradient[0] / mcCycleCount;
        gradients[i][1] = gradient[1] / mcCycleCount;
    }

    std::vector<std::vector<double>> energyGrads(walkerCount, std::vector<double>(2, 0));

    for (size_t i = 0; i < walkerCount; i++)
    {
        energyGrads[i][0] = 2 * (gradMuls[i][0] - gradients[i][0] * energies[i]);
        energyGrads[i][1] = 2 * (gradMuls[i][1] - gradients[i][1] * energies[i]);
    }

    /*
    both calcMeanStd and vectorCalcMeanStd calculate the standard deviation
    of the samples, not the standard deviation of the means. This means that
    the uncertanties are sqrt(walkerCount) too large. I did not notice this
    until I had ran all the calculations, so instead of fixing it here,
    I fixed it in the plotting program.
    */
    auto energyTouple = calcMeanStd(energies);
    auto gradTouple = vectorCalcMeanStd(energyGrads);

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

    return {
        std::get<0>(energyTouple),
        std::get<1>(energyTouple),
        std::get<0>(gradTouple),
        std::get<1>(gradTouple),
    };
}
