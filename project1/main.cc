#include <memory>
#include <iostream>
#include <fstream>
#include <string>

#include "alphaSamplers.hh"
#include "onebody.hh"

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

void multiSampleSaver(
    const MCMode mode, double magnitude, size_t mcCycleCount, size_t walkerCount)
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
    size_t cycleCount1 = std::pow(2, 1 + (size_t)std::log2(mcCycleCount));
    size_t cycleCount100 = std::pow(2, 1 + (size_t)std::log2(mcCycleCount / 10));
    size_t cycleCount500 = std::pow(2, 1 + (size_t)std::log2(mcCycleCount / std::sqrt(500)));

    mcSampler<1, 1>(
        mode, magnitude, cycleCount1, cycleCount1 / 100, walkerCount,
        args.a, args.stateSize, 0.4, true, "data/samples/d1N1_" + modeStr);
    mcSampler<100, 2>(
        mode, magnitude, cycleCount100, cycleCount100 / 100, walkerCount,
        args.a, args.stateSize, 0.4, true, "data/samples/d2N100_" + modeStr);
    mcSampler<500, 3>(
        mode, magnitude, cycleCount500, cycleCount500 / 100, walkerCount,
        args.a, args.stateSize, 0.4, true, "data/samples/d3N500_" + modeStr);
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

void elipticalMultiSampleSaver(
    const MCMode mode, double magnitude, size_t mcCycleCount, size_t walkerCount)
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

    SamplerArgs args{
        .a = 0.0043,
        .stateSize = 1,
        .beta = 2.82843,
        .gamma = 2.82843};

    size_t cycleCount1 = std::pow(2, 1 + (size_t)std::log2(mcCycleCount));
    size_t cycleCount100 = std::pow(2, 1 + (size_t)std::log2(mcCycleCount / 10));
    size_t cycleCount500 = std::pow(2, 1 + (size_t)std::log2(mcCycleCount / std::sqrt(500)));

    std::vector<double> opt_alpha_list = {0.4976805925444665, 0.4900523835748852, 0.4845254493116315};

    mcSampler<10>(
        mode, magnitude, cycleCount1, cycleCount1 / 100, walkerCount, args.a, args.stateSize,
        opt_alpha_list[0], args.beta, args.gamma, true, "data/samples/eliptical_d3N10_" + modeStr);
    mcSampler<50>(
        mode, magnitude, cycleCount100, cycleCount100 / 100, walkerCount, args.a, args.stateSize,
        opt_alpha_list[1], args.beta, args.gamma, true, "data/samples/eliptical_d3N50_" + modeStr);
    mcSampler<100>(
        mode, magnitude, cycleCount500, cycleCount500 / 100, walkerCount, args.a, args.stateSize,
        opt_alpha_list[2], args.beta, args.gamma, true, "data/samples/eliptical_d3N100_" + modeStr);
}

void multiOnebodyCalculator(double optimalTimeStep, size_t mcCycleCount, size_t walkercount)
{
    SamplerArgs args{
        .a = 0.0043,
        .stateSize = 1,
        .beta = 2.82843,
        .gamma = 2.82843};
    std::vector<double> opt_alpha_list = {0.4976805925444665, 0.4900523835748852, 0.4845254493116315};

    Timer timer;
    size_t cyclecount;

    cyclecount = mcCycleCount / std::sqrt(10);
    onebodyCalculator<10>(
        MCMode::METHAS, optimalTimeStep, cyclecount, cyclecount / 100, walkercount, 0,
        args.stateSize, opt_alpha_list[0], args.beta, args.gamma, "data/onebody/N10_noJastrow.dat");
    std::cerr << timer.get_pretty() << "\n";
    timer.restart();
    onebodyCalculator<10>(
        MCMode::METHAS, optimalTimeStep, cyclecount, cyclecount / 100, walkercount, args.a,
        args.stateSize, opt_alpha_list[0], args.beta, args.gamma, "data/onebody/N10_Jastrow.dat");
    std::cerr << timer.get_pretty() << "\n";
    timer.restart();

    cyclecount = mcCycleCount / std::sqrt(50);
    onebodyCalculator<50>(
        MCMode::METHAS, optimalTimeStep, cyclecount, cyclecount / 100, walkercount, 0,
        args.stateSize, opt_alpha_list[1], args.beta, args.gamma, "data/onebody/N50_noJastrow.dat");
    std::cerr << timer.get_pretty() << "\n";
    timer.restart();
    onebodyCalculator<50>(
        MCMode::METHAS, optimalTimeStep, cyclecount, cyclecount / 100, walkercount, args.a,
        args.stateSize, opt_alpha_list[1], args.beta, args.gamma, "data/onebody/N50_Jastrow.dat");
    std::cerr << timer.get_pretty() << "\n";
    timer.restart();

    cyclecount = mcCycleCount / std::sqrt(100);
    onebodyCalculator<100>(
        MCMode::METHAS, optimalTimeStep, cyclecount, cyclecount / 100, walkercount, 0,
        args.stateSize, opt_alpha_list[2], args.beta, args.gamma, "data/onebody/N100_noJastrow.dat");
    std::cerr << timer.get_pretty() << "\n";
    timer.restart();
    onebodyCalculator<100>(
        MCMode::METHAS, optimalTimeStep, cyclecount, cyclecount / 100, walkercount, args.a,
        args.stateSize, opt_alpha_list[2], args.beta, args.gamma, "data/onebody/N100_Jastrow.dat");
    std::cerr << timer.get_pretty() << "\n";
    timer.restart();
}

int main()
{
    std::vector<double> alphaVec = linspace(0.25, 0.75, 51);

    size_t mcCycleCount = 1e4;
    size_t walkerCount = 8;

    // multiCalibrator(alphaVec, MCMode::MET, mcCycleCount, 128);
    // multiCalibrator(alphaVec, MCMode::METHAS, mcCycleCount, 128);

    // double optimalStepSize = 4.125;
    // multiSampler<1>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);
    // multiSampler<2>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);
    // multiSampler<3>(alphaVec, MCMode::MET, optimalStepSize, mcCycleCount, walkerCount);

    double optimalTimeStep = 0.48828;
    // multiSampler<1>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);
    // multiSampler<2>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);
    // multiSampler<3>(alphaVec, MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);

    // double maxSteps = 100;
    // double sphericalLR = 0.02;
    // gradMultiSampler(
    //     0.4, sphericalLR, maxSteps, MCMode::METHAS,
    //     optimalTimeStep, mcCycleCount, walkerCount);
    // multiSampleSaver(MCMode::METHAS, optimalTimeStep, mcCycleCount, walkerCount);

    // double elipticalLR = 0.0004;
    // elipticalMultiSampler(
    //     0.5, elipticalLR, maxSteps, MCMode::METHAS,
    //     optimalTimeStep, mcCycleCount * 10, walkerCount);
    // elipticalMultiSampleSaver(MCMode::METHAS, optimalTimeStep, mcCycleCount * 10, walkerCount);

    multiOnebodyCalculator(optimalTimeStep, mcCycleCount*10, 128);

    return 0;
}
