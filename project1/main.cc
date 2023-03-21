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
    std::string s1 = "data/d" + std::to_string(d) + "N";
    std::string s2 = "_" + modeStr + ".dat";

    SamplerArgs args;

    alphaSampler<1, d>(alphaVec, mode, args, mcCycleCount, walkerCount, s1 + "1" + s2);
    alphaSampler<10, d>(alphaVec, mode, args, mcCycleCount, walkerCount, s1 + "10" + s2);
    alphaSampler<100, d>(alphaVec, mode, args, mcCycleCount, walkerCount, s1 + "100" + s2);
    alphaSampler<500, d>(alphaVec, mode, args, mcCycleCount, walkerCount, s1 + "500" + s2);
}

void gradMultiSampler(
    double alpha0, double learningRate, size_t maxSteps,
    const MCMode mode, size_t mcCycleCount, size_t walkerCount)
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
        alpha0, learningRate, maxSteps, mode, args,
        mcCycleCount, walkerCount, "data/grad_d1N1_" + modeStr);
    gradAlphaSampler<500, 3>(
        alpha0, learningRate, maxSteps, mode, args,
        mcCycleCount, walkerCount, "data/grad_d3N500_" + modeStr);
}

void elipticalMultiSampler(
    double alpha0, double learningRate, size_t maxSteps,
    const MCMode mode, size_t mcCycleCount, size_t walkerCount)
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
    std::string s1 = "data/eliptical_d3N";
    std::string s2 = "_" + modeStr + ".dat";

    SamplerArgs args{
        .a = 0.0043,
        .stateSize = 1,
        .beta = 2.82843,
        .gamma = 2.82843};

    gradAlphaSampler<10, 3>(
        alpha0, learningRate, maxSteps, mode, args,
        mcCycleCount, walkerCount, s1 + "10" + s2);
    gradAlphaSampler<50, 3>(
        alpha0, learningRate, maxSteps, mode, args,
        mcCycleCount, walkerCount, s1 + "50" + s2);
    gradAlphaSampler<100, 3>(
        alpha0, learningRate, maxSteps, mode, args,
        mcCycleCount, walkerCount, s1 + "100" + s2);
}

int main()
{
    std::vector<double> alphaVec = linspace(0.25, 0.75, 51);

    size_t fullCycleCount = 1e3;
    size_t walkerCount = 8;

    multiCalibrator(alphaVec, MCMode::MET, fullCycleCount, walkerCount);
    multiCalibrator(alphaVec, MCMode::METHAS, fullCycleCount, walkerCount);

    // multiSampler<1>(alphaVec, MCMode::MET, fullCycleCount, walkerCount);
    // multiSampler<2>(alphaVec, MCMode::MET, fullCycleCount, walkerCount);
    // multiSampler<3>(alphaVec, MCMode::MET, fullCycleCount, walkerCount);

    // multiSampler<1>(alphaVec, MCMode::METHAS, fullCycleCount, walkerCount);
    // multiSampler<2>(alphaVec, MCMode::METHAS, fullCycleCount, walkerCount);
    // multiSampler<3>(alphaVec, MCMode::METHAS, fullCycleCount, walkerCount);

    size_t gradCycleCount = 1e5;
    double maxSteps = 50;

    // gradMultiSampler(0.4, 0.02, maxSteps, MCMode::METHAS, gradCycleCount, walkerCount);
    // elipticalMultiSampler(0.5, 0.0001, maxSteps, MCMode::METHAS, gradCycleCount, walkerCount);

    return 0;
}
