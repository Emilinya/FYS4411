#pragma once

#include "methasMCSampler.hh"
#include "ParticleSystem.hh"
#include "WaveFunction.hh"
#include "metMCSampler.hh"
#include "SphericalWF.hh"
#include "Particle.hh"
#include "Random.hh"
#include "utils.hh"

template <size_t N, size_t d>
void methasSampler(
    std::vector<double> &alphaVec, double timeStep, size_t mcCycleCount,
    size_t walkerCount, std::string filename)
{
    size_t trueCycleCount = mcCycleCount / sqrt(N);
    size_t burnCycleCount = trueCycleCount / 100;

    double mass = 1;
    double diameter = 0;
    double stateSize = 1;

    std::cout << "N=" << N << ", d=" << d << std::endl;
    std::ofstream dataFile;
    dataFile.precision(14);
    dataFile.open(filename);
    for (size_t i = 0; i < alphaVec.size(); i++)
    {
        rprint("  " << i + 1 << "/" << alphaVec.size());

        auto [energy, std] = methasMonteCarloSampler<N, d, SphericalWF<N, d>>(
            timeStep, trueCycleCount, burnCycleCount, walkerCount, diameter, mass, stateSize, alphaVec[i]);

        dataFile << alphaVec[i] << " " << energy << " " << std << "\n";
    }
    std::cout << std::endl;
    dataFile.close();
}

template <size_t N, size_t d>
void metSampler(
    std::vector<double> &alphaVec, double stepSize, size_t mcCycleCount,
    size_t walkerCount, std::string filename)
{
    size_t trueCycleCount = mcCycleCount / sqrt(N);
    size_t burnCycleCount = trueCycleCount / 100;

    double mass = 1;
    double diameter = 0;
    double stateSize = 1;

    std::cout << "N=" << N << ", d=" << d << std::endl;
    std::ofstream dataFile;
    dataFile.precision(14);
    dataFile.open(filename);
    for (size_t i = 0; i < alphaVec.size(); i++)
    {
        rprint("  " << i + 1 << "/" << alphaVec.size());

        auto [energy, std] = metMonteCarloSampler<N, d, SphericalWF<N, d>>(
            stepSize, trueCycleCount, burnCycleCount, walkerCount, diameter, mass, stateSize, alphaVec[i]);

        dataFile << alphaVec[i] << " " << energy << " " << std << "\n";
    }
    std::cout << std::endl;
    dataFile.close();
}

template <size_t N, size_t d>
double calibrateStepSize(double alpha, double analVal, size_t mcCycleCount, size_t walkerCount)
{
    size_t trueCycleCount = mcCycleCount / sqrt(N);
    size_t burnCycleCount = trueCycleCount / 100;
    double initialStepSize = 0.5;

    auto getDiff = [&](double stepSize)
    {
        auto mcTouple = methasMonteCarloSampler<N, d, SphericalWF<N, d>>(
            stepSize, trueCycleCount, burnCycleCount, walkerCount, 0, 1, 1, alpha);
        return abs(analVal - std::get<0>(mcTouple));
    };

    double initialDiff = getDiff(initialStepSize);
    double halfDiff = getDiff(0.5 * initialStepSize);
    double doubleDiff = getDiff(2.0 * initialStepSize);

    double diff;
    double left;
    double right;
    bool didDouble;

    if (initialDiff < halfDiff && initialDiff < doubleDiff)
    {
        // The initial step size was optimal
        return initialStepSize;
    }
    else if (halfDiff < doubleDiff)
    {
        // halving it improved the result, try halving it more
        double stepSize = 0.5 * initialStepSize;
        diff = halfDiff;
        while (true) {
            double newStepSize = stepSize * 0.5;
            double newDiff = getDiff(newStepSize);

            if (newDiff < diff) {
                stepSize = newStepSize;
                diff = newDiff;
            } else {
                break;
            }
        }

        // We now know the region where the optimal value lies
        left = stepSize / 2;
        right = stepSize;
        didDouble = false;
    } else {
        // doubling it improved the result, try doubling it more
        double stepSize = 2. * initialStepSize;
        diff = doubleDiff;
        while (true)
        {
            double newStepSize = stepSize * 2.;
            double newDiff = getDiff(newStepSize);

            if (newDiff < diff)
            {
                stepSize = newStepSize;
                diff = newDiff;
            }
            else
            {
                break;
            }
        }

        left = stepSize;
        right = stepSize * 2.;
        didDouble = true;
    }

    // We now know the region where the optimal value lies, find it using binary search

    double mid = (left + right) * 0.5;
    double midDiff = getDiff(mid);

    while (right - left > 1e-3) {
        if (midDiff < diff) {
            double q1 = (left + mid) * 0.5;
            double q3 = (mid + right) * 0.5;

            double q1Diff = getDiff(q1);
            double q3Diff = getDiff(q3);

            if (midDiff < q1Diff  && midDiff < q3Diff) {
                print(mid, midDiff);
                return mid;
            }
            else if (q1Diff < q3Diff) {
                right = mid;
                mid = q1;
                midDiff = q1Diff;
            }
        } else {
            if (didDouble) {
                // left is better
                right = mid;
            } else {
                // right is better
                left = mid;
            }
            mid = (left + right) * 0.5;
            midDiff = getDiff(mid);
        }
    }
    print(mid, midDiff);

    return mid;
}
