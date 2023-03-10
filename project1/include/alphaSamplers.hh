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
        dataFile << out.gradE[0] << " " << out.stdGradE[0] << "\n";
    }
    std::cout << std::endl;
    dataFile.close();
}

template <size_t N, size_t d>
void gradAlphaSampler(
    double alpha0, double learningRate, size_t maxSteps, const MCMode mode,
    double magnitude, size_t mcCycleCount, size_t walkerCount, std::string filename)
{
    size_t cycleCount = mcCycleCount / sqrt(N) / 100;

    double mass = 1;
    double diameter = 0;
    double stateSize = 1;

    std::cout << "N=" << N << ", d=" << d << std::endl;
    std::ofstream dataFile;
    dataFile.precision(14);
    dataFile.open(filename);

    double moment = 0;
    double decay = 0.9;

    double grad = std::nan("");
    double alpha = alpha0;
    for (size_t i = 0; i < maxSteps; i++)
    {
        rprint("  "
               << "i = " << i + 1 << ", alpha = "
               << alpha << ", grad = " << grad);

        MCSamplerOut out = mcSampler<N, d, SphericalWF<N, d>>(
            mode, magnitude, cycleCount, cycleCount / 100, walkerCount,
            diameter, mass, stateSize, alpha);
        grad = out.gradE[0];

        dataFile << alpha << " " << out.E << " " << out.stdE << "\n";

        if (std::abs(grad) < 1e-14)
        {
            break;
        }

        moment = decay * moment + (1 - decay) * grad * grad;
        alpha = alpha - learningRate * grad / (std::sqrt(moment) + 1e-14);

        cycleCount = std::min(cycleCount*100, (size_t)(cycleCount*1.05));
    }

    std::cout << std::endl;
    dataFile.close();
}

template <size_t N, size_t d>
double calibrateMagnitude(
    double alpha, double analVal, const MCMode mode,
    size_t mcCycleCount, size_t walkerCount, std::string filename)
{
    size_t trueCycleCount = mcCycleCount / sqrt(N);
    size_t burnCycleCount = trueCycleCount / 100;
    double initialMagnitude = 0.5;

    std::ofstream dataFile;
    dataFile.precision(14);
    dataFile.open(filename);
    std::cout << "N=" << N << ", d=" << d << std::endl;

    auto getDiff = [&](double magnitude)
    {
        print(" ", magnitude);
        MCSamplerOut out = mcSampler<N, d, SphericalWF<N, d>>(
            mode, magnitude, trueCycleCount, burnCycleCount, walkerCount, 0, 1, 1, alpha);
        dataFile << magnitude << " " << abs(analVal - out.E) << " " << out.stdGradE[0] << "\n";
        return abs(analVal - out.E);
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
        dataFile.close();
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
                dataFile.close();
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

    dataFile.close();
    return mid;
}
