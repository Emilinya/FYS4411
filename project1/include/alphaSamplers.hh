#pragma once

#include "ParticleSystem.hh"
#include "WaveFunction.hh"
#include "SphericalWF.hh"
#include "mcSampler.hh"
#include "Particle.hh"
#include "Random.hh"
#include "utils.hh"

template <size_t N, size_t d>
void alphaSampler(
    std::vector<double> &alphaVec, const MCMode mode, double magnitude,
    size_t mcCycleCount, size_t walkerCount, std::string filename)
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

        MCSamplerOut out = mcSampler<N, d, SphericalWF<N, d>>(
            mode, magnitude, trueCycleCount, burnCycleCount, walkerCount,
            diameter, mass, stateSize, alphaVec[i]);

        dataFile << alphaVec[i] << " " << out.E << " " << out.stdE << " ";
        dataFile << out.logGrad[0] << " " << out.stdLogGrad[0] << "\n";
    }
    std::cout << std::endl;
    dataFile.close();
}

template <size_t N, size_t d>
double calibrateMagnitude(
    double alpha, double analVal, const MCMode mode,
    size_t mcCycleCount, size_t walkerCount)
{
    size_t trueCycleCount = mcCycleCount / sqrt(N);
    size_t burnCycleCount = trueCycleCount / 100;
    double initialMagnitude = 0.5;

    auto getDiff = [&](double magnitude)
    {
        auto mcTouple = mcSampler<N, d, SphericalWF<N, d>>(
            mode, magnitude, trueCycleCount, burnCycleCount, walkerCount, 0, 1, 1, alpha);
        return abs(analVal - std::get<0>(mcTouple));
    };

    double initialDiff = getDiff(initialMagnitude);
    double halfDiff = getDiff(0.5 * initialMagnitude);
    double doubleDiff = getDiff(2.0 * initialMagnitude);

    double diff;
    double left;
    double right;
    bool didDouble;

    if (initialDiff < halfDiff && initialDiff < doubleDiff)
    {
        // The initial magnitude was optimal
        return initialMagnitude;
    }
    else if (halfDiff < doubleDiff)
    {
        // halving it improved the result, try halving it more
        double magnitude = 0.5 * initialMagnitude;
        diff = halfDiff;
        while (true)
        {
            double newMagnitude = magnitude * 0.5;
            double newDiff = getDiff(newMagnitude);

            if (newDiff < diff)
            {
                magnitude = newMagnitude;
                diff = newDiff;
            }
            else
            {
                break;
            }
        }

        // We now know the region where the optimal value lies
        left = magnitude / 2;
        right = magnitude;
        didDouble = false;
    }
    else
    {
        // doubling it improved the result, try doubling it more
        double magnitude = 2. * initialMagnitude;
        diff = doubleDiff;
        while (true)
        {
            double newMagnitude = magnitude * 2.;
            double newDiff = getDiff(newMagnitude);

            if (newDiff < diff)
            {
                magnitude = newMagnitude;
                diff = newDiff;
            }
            else
            {
                break;
            }
        }

        left = magnitude;
        right = magnitude * 2.;
        didDouble = true;
    }

    // We now know the region where the optimal value lies, find it using binary search

    double mid = (left + right) * 0.5;
    double midDiff = getDiff(mid);

    while (right - left > 1e-3)
    {
        if (midDiff < diff)
        {
            double q1 = (left + mid) * 0.5;
            double q3 = (mid + right) * 0.5;

            double q1Diff = getDiff(q1);
            double q3Diff = getDiff(q3);

            if (midDiff < q1Diff && midDiff < q3Diff)
            {
                return mid;
            }
            else if (q1Diff < q3Diff)
            {
                right = mid;
                mid = q1;
                midDiff = q1Diff;
            }
        }
        else
        {
            if (didDouble)
            {
                // left is better
                right = mid;
            }
            else
            {
                // right is better
                left = mid;
            }
            mid = (left + right) * 0.5;
            midDiff = getDiff(mid);
        }
    }

    return mid;
}