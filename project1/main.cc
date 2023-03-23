#include <memory>
#include <iostream>
#include <fstream>
#include <string>

#include "alphaSamplers.hh"

void multiCalibrator(
    std::vector<double> &alphaVec, const MCMode mode,
    size_t mcCycleCount, size_t walkerCount)
{
    std::string modeStr;
    if (mode == MCMode::MET)
    {
        modeStr = "met";
    }
    else
    {
        modeStr = "methas";
    }

    calibrateMagnitude<1, 1>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d1N1_" + modeStr + ".dat");
    calibrateMagnitude<1, 2>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d2N1_" + modeStr + ".dat");
    calibrateMagnitude<1, 3>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d3N1_" + modeStr + ".dat");

    calibrateMagnitude<10, 1>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d1N10_" + modeStr + ".dat");
    calibrateMagnitude<10, 2>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d2N10_" + modeStr + ".dat");
    calibrateMagnitude<10, 3>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d3N10_" + modeStr + ".dat");

    calibrateMagnitude<100, 1>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d1N100_" + modeStr + ".dat");
    calibrateMagnitude<100, 2>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d2N100_" + modeStr + ".dat");
    calibrateMagnitude<100, 3>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d3N100_" + modeStr + ".dat");

    calibrateMagnitude<500, 1>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d1N500_" + modeStr + ".dat");
    calibrateMagnitude<500, 2>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d2N500_" + modeStr + ".dat");
    calibrateMagnitude<500, 3>(
        alphaVec, mode, mcCycleCount,
        walkerCount, "data/calibrate/d3N500_" + modeStr + ".dat");
}

template <size_t d>
void multiSampler(
    std::vector<double> &alphaVec, const MCMode mode,
    double magnitude, size_t mcCycleCount, size_t walkerCount)
{
    std::string modeStr;
    if (mode == MCMode::MET)
    {
        modeStr = "met";
    }
    else
    {
        modeStr = "methas";
    }
    std::string s1 = "data/full/d" + std::to_string(d) + "N";
    std::string s2 = "_" + modeStr + ".dat";

    SamplerArgs args;
    alphaSampler<1, d>(alphaVec, mode, magnitude, args, mcCycleCount, walkerCount, s1 + "1" + s2);
    alphaSampler<10, d>(alphaVec, mode, magnitude, args, mcCycleCount, walkerCount, s1 + "10" + s2);
    alphaSampler<100, d>(alphaVec, mode, magnitude, args, mcCycleCount, walkerCount, s1 + "100" + s2);
    alphaSampler<500, d>(alphaVec, mode, magnitude, args, mcCycleCount, walkerCount, s1 + "500" + s2);
}

void gradMultiSampler(
    double alpha0, double learningRate, size_t maxSteps, const MCMode mode,
    double magnitude, size_t mcCycleCount, size_t walkerCount)
{
    std::string modeStr;
    if (mode == MCMode::MET)
    {
        modeStr = "met.dat";
    }
    else
    {
        modeStr = "methas.dat";
    }

    SamplerArgs args;
    gradAlphaSampler<1, 1>(
        alpha0, learningRate, maxSteps, mode, magnitude, args,
        mcCycleCount, walkerCount, "data/grad/d1N1_" + modeStr);
    gradAlphaSampler<100, 2>(
        alpha0, learningRate, maxSteps, mode, magnitude, args,
        mcCycleCount, walkerCount, "data/grad/d2N100_" + modeStr);
    gradAlphaSampler<500, 3>(
        alpha0, learningRate, maxSteps, mode, magnitude, args,
        mcCycleCount, walkerCount, "data/grad/d3N500_" + modeStr);
}

void elipticalMultiSampler(
    double alpha0, double learningRate, size_t maxSteps, const MCMode mode,
    double magnitude, size_t mcCycleCount, size_t walkerCount)
{
    std::string modeStr;
    if (mode == MCMode::MET)
    {
        modeStr = "met";
    }
    else
    {
        modeStr = "methas";
    }
    std::string s1 = "data/grad/eliptical_d3N";
    std::string s2 = "_" + modeStr + ".dat";

    SamplerArgs args{
        .a = 0.0043,
        .stateSize = 1,
        .beta = 2.82843,
        .gamma = 2.82843};

    gradAlphaSampler<10, 3>(
        alpha0, learningRate, maxSteps, mode, magnitude, args,
        mcCycleCount, walkerCount, s1 + "10" + s2);
    gradAlphaSampler<50, 3>(
        alpha0, learningRate, maxSteps, mode, magnitude, args,
        mcCycleCount, walkerCount, s1 + "50" + s2);
    gradAlphaSampler<100, 3>(
        alpha0, learningRate, maxSteps, mode, magnitude, args,
        mcCycleCount, walkerCount, s1 + "100" + s2);
}

int main()
{
    std::vector<double> alphaVec = linspace(0.25, 0.75, 51);

    size_t mcCycleCount = 1e4;
    size_t walkerCount = 8;

    multiCalibrator(alphaVec, MCMode::MET, mcCycleCount, 128);
    multiCalibrator(alphaVec, MCMode::METHAS, mcCycleCount, 128);

    double optimalStepSize = 4.125;
    multiSampler<1>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);
    multiSampler<2>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);
    multiSampler<3>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);

    double optimalTimeStep = 0.48828;
    multiSampler<1>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);
    multiSampler<2>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);
    multiSampler<3>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);

    double maxSteps = 100;
    double sphericalLR = 0.02;
    gradMultiSampler(
        0.4, sphericalLR, maxSteps, MCMode::METHAS,
        optimalTimeStep, mcCycleCount, walkerCount);

    double elipticalLR = 0.0004;
    elipticalMultiSampler(
        0.5, elipticalLR, maxSteps, MCMode::METHAS,
        optimalTimeStep, mcCycleCount * 10, walkerCount);

    return 0;
}
