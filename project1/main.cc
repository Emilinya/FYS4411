#include <memory>
#include <iostream>
#include <fstream>
#include <string>

#include "samplers.hh"

template <size_t d>
void metMultiSampler(std::vector<double> &alphaVec, double stepSize, size_t mcCycleCount, size_t walkerCount)
{
    std::string dStr = std::to_string(d);
    metSampler<1, d>(alphaVec, stepSize, mcCycleCount, walkerCount, "data/d" + dStr + "N1_met.dat");
    metSampler<10, d>(alphaVec, stepSize, mcCycleCount, walkerCount, "data/d" + dStr + "N10_met.dat");
    metSampler<100, d>(alphaVec, stepSize, mcCycleCount, walkerCount, "data/d" + dStr + "N100_met.dat");
    metSampler<500, d>(alphaVec, stepSize, mcCycleCount, walkerCount, "data/d" + dStr + "N500_met.dat");
}

template <size_t d>
void methasMultiSampler(std::vector<double> &alphaVec, double timeStep, size_t mcCycleCount, size_t walkerCount)
{
    std::string dStr = std::to_string(d);
    methasSampler<1, d>(alphaVec, timeStep, mcCycleCount, walkerCount, "data/d" + dStr + "N1_methas.dat");
    methasSampler<10, d>(alphaVec, timeStep, mcCycleCount, walkerCount, "data/d" + dStr + "N10_methas.dat");
    methasSampler<100, d>(alphaVec, timeStep, mcCycleCount, walkerCount, "data/d" + dStr + "N100_methas.dat");
    methasSampler<500, d>(alphaVec, timeStep, mcCycleCount, walkerCount, "data/d" + dStr + "N500_methas.dat");
}

int main()
{
    std::vector<double> alphaVec = linspace(0.25, 0.75, 51);
    size_t mcCycleCount = 1e4;
    size_t walkerCount = 8;

    // calibrateStepSize<100, 1>(0.4, 5.125 * 10, 1e7, 8);
    // calibrateStepSize<10, 1>(0.4, 5.125, 1e7, 8);
    // calibrateStepSize<1, 1>(0.4, 5.125 * 0.1, 1e7, 8);
    
    double optimalStepSize = 0.1;
    metMultiSampler<1>(alphaVec, optimalStepSize, mcCycleCount, walkerCount);
    // metMultiSampler<2>(alphaVec, optimalStepSize, mcCycleCount, walkerCount);
    // metMultiSampler<3>(alphaVec, optimalStepSize, mcCycleCount, walkerCount);

    double optimalTimeStep = 0.005;
    methasMultiSampler<1>(alphaVec, optimalTimeStep, mcCycleCount, walkerCount);

    return 0;
}
