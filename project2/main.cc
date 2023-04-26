#include <memory>
#include <iostream>
#include <fstream>
#include <string>

#include "testRBMWF.hh"
#include "energyMinimizer.hh"

template <size_t N, size_t d, size_t M>
void minAndSample(
    double learningRate, size_t maxSteps, const MCMode mode, double magnitude,
    double sigma, bool interactions, size_t mcCycleCount, size_t walkerCount,
    std::string folder)
{
    std::string modeStr;
    if (mode == MCMode::MET) {
        modeStr = "met";
    } else {
        modeStr = "methas";
    }

    char buffer[50];
    snprintf(buffer, 50, "data/%s/N%ldd%ldM%ld_%s", folder.c_str(), N, d, M, modeStr.c_str());
    std::string str(buffer);

    auto [_, params] = minimizer<N, d, M>(
        learningRate, maxSteps, mode, magnitude, sigma, interactions, mcCycleCount, walkerCount, str + "_grad.dat");

    size_t samplerCycles = std::pow(2, 1 + (size_t)std::log2(10 * mcCycleCount));
    mcSampler<N, d, M>(
        mode, magnitude, samplerCycles, samplerCycles / 100,
        walkerCount, sigma, interactions, params, true, str + "_samples.dat");
}

void MComp(
    double learningRate, size_t maxSteps, const MCMode mode, double magnitude,
    double sigma, size_t mcCycleCount, size_t walkerCount)
{
    minAndSample<1, 1, 1>(learningRate, maxSteps, mode, magnitude, sigma, false, mcCycleCount, walkerCount, "MComp");
    minAndSample<1, 1, 2>(learningRate, maxSteps, mode, magnitude, sigma, false, mcCycleCount, walkerCount, "MComp");
    minAndSample<1, 1, 5>(learningRate, maxSteps, mode, magnitude, sigma, false, mcCycleCount, walkerCount, "MComp");
    minAndSample<1, 1, 10>(learningRate, maxSteps, mode, magnitude, sigma, false, mcCycleCount, walkerCount, "MComp");

    minAndSample<2, 3, 2>(learningRate, maxSteps, mode, magnitude, sigma, false, mcCycleCount, walkerCount, "MComp");
    minAndSample<2, 3, 5>(learningRate, maxSteps, mode, magnitude, sigma, false, mcCycleCount, walkerCount, "MComp");
}

void lrComp(
    size_t maxSteps, const MCMode mode, double magnitude,
    double sigma, size_t mcCycleCount, size_t walkerCount)
{
    std::string modeStr;
    if (mode == MCMode::MET) {
        modeStr = "met";
    } else {
        modeStr = "methas";
    }

    arma::vec lrs = arma::logspace(-3, 0, 10);
    for (double lr : lrs)
    {
        char buffer[50];
        snprintf(buffer, 50, "data/lrComp/N1d1M10_lr=%.5f_%s.dat", lr, modeStr.c_str());
        std::string filename(buffer);
        minimizer<1, 1, 10>(
            lr, maxSteps, mode, magnitude, sigma, false, mcCycleCount, walkerCount, filename);
    }
}

void interactions(
    double learningRate, size_t maxSteps, const MCMode mode, double magnitude,
    double sigma, size_t mcCycleCount, size_t walkerCount)
{
    minAndSample<2, 2, 1>(
        learningRate, maxSteps, mode, magnitude, sigma, true, mcCycleCount, walkerCount, "interactions");
    minAndSample<2, 2, 2>(
        learningRate, maxSteps, mode, magnitude, sigma, true, mcCycleCount, walkerCount, "interactions");
    minAndSample<2, 2, 5>(
        learningRate, maxSteps, mode, magnitude, sigma, true, mcCycleCount, walkerCount, "interactions");
    minAndSample<2, 2, 10>(
        learningRate, maxSteps, mode, magnitude, sigma, true, mcCycleCount, walkerCount, "interactions");
}

    int main()
{
    // testRBMWF();

    double sigma = 1;

    size_t descentSteps = 1000;
    size_t walkerCount = 8;
    size_t mcCycles = 1e5;
    double optimalLR = 0.00215;

    // safety exit - we don't want to override results
    exit(1);

    double optimalStepsize = 4.125;
    lrComp(descentSteps, MCMode::MET, optimalStepsize, sigma, mcCycles, walkerCount);
    MComp(optimalLR, descentSteps, MCMode::MET, optimalStepsize, sigma, mcCycles, walkerCount);
    interactions(optimalLR, descentSteps, MCMode::MET, optimalStepsize, sigma, mcCycles, walkerCount);

    double optimalTimestep = 0.48828;
    lrComp(descentSteps, MCMode::METHAS, optimalTimestep, sigma, mcCycles, walkerCount);
    MComp(optimalLR, descentSteps, MCMode::METHAS, optimalTimestep, sigma, mcCycles, walkerCount);
    interactions(optimalLR, descentSteps, MCMode::METHAS, optimalTimestep, sigma, mcCycles, walkerCount);

    return 0;
}
