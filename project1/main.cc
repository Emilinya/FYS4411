#include <memory>
#include <iostream>
#include <fstream>
#include <string>

#include "alphaSamplers.hh"

template <size_t d>
void multiSampler(std::vector<double> &alphaVec, const MCMode mode, double magnitude, size_t mcCycleCount, size_t walkerCount)
{
    std::string modeStr;
    if (mode == MCMode::MET) {
        modeStr = "met";
    }
    else
    {
        modeStr = "methas";
    }
    std::string s1 = "data/d" + std::to_string(d) + "N";
    std::string s2 = "_" + modeStr + ".dat";

    alphaSampler<1, d>(alphaVec, mode, magnitude, mcCycleCount, walkerCount, s1 + "1" + s2);
    // alphaSampler<10, d>(alphaVec, mode, magnitude, mcCycleCount, walkerCount, s1 + "10" + s2);
    // alphaSampler<100, d>(alphaVec, mode, magnitude, mcCycleCount, walkerCount, s1 + "100" + s2);
    // alphaSampler<500, d>(alphaVec, mode, magnitude, mcCycleCount, walkerCount, s1 + "500" + s2);
}

int main()
{
    std::vector<double> alphaVec = linspace(0.25, 0.75, 51);
    size_t mcCycleCount = 5e5;
    size_t walkerCount = 8;

    // calibrateStepSize<100, 1>(0.4, 5.125 * 10, 1e7, 8);
    // calibrateStepSize<10, 1>(0.4, 5.125, 1e7, 8);
    // calibrateStepSize<1, 1>(0.4, 5.125 * 0.1, 1e7, 8);
    
    double optimalStepSize = 0.1;
    multiSampler<1>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);
    // multiSampler<2>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);
    // multiSampler<3>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);

    double optimalTimeStep = 0.005;
    multiSampler<1>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);
    // multiSampler<2>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);
    // multiSampler<3>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);

    // methasSampler<100, 1>(alphaVec, optimalTimeStep, 1e4, walkerCount, "data/temp.dat");

    return 0;
}
