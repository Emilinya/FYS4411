#pragma once

#include "ParticleSystem.hh"
#include "WaveFunction.hh"
#include "SphericalWF.hh"
#include "mcSampler.hh"
#include "Particle.hh"
#include "Random.hh"
#include "utils.hh"

#include <string.h>

struct SamplerArgs
{
    double a = 0;
    double stateSize = 1;
    double beta = std::nan("");
    double gamma = std::nan("");
};

// calculate energy and gradient for alpha values in alphaVec, and save to file
template <size_t N, size_t d>
void alphaSampler(
    std::vector<double> &alphaVec, const MCMode mode, double magnitude,
    SamplerArgs args, size_t mcCycleCount, size_t walkerCount, std::string filename)
{
    size_t cycleCount = mcCycleCount / std::sqrt(N);

    std::cout << "N=" << N << ", d=" << d << std::endl;
    std::ofstream dataFile(filename);
    if (!dataFile)
    {
        std::cerr << "Error opening " + filename + ": " << strerror(errno) << std::endl;
        exit(0);
    }

    dataFile.precision(14);
    for (size_t i = 0; i < alphaVec.size(); i++)
    {
        rprint("  " << i + 1 << "/" << alphaVec.size());

        MCSamplerOut out;
        if (std::isnan(args.beta))
        {
            out = mcSampler<N, d>(
                mode, magnitude, cycleCount, cycleCount / 100,
                walkerCount, args.a, args.stateSize, alphaVec[i]);
        }
        else
        {
            out = mcSampler<N>(
                mode, magnitude, cycleCount, cycleCount / 100,
                walkerCount, args.a, args.stateSize, alphaVec[i], args.beta, args.gamma);
        }

        dataFile << alphaVec[i] << " " << out.E << " " << out.stdE << " ";
        dataFile << out.gradE[0] << " " << out.stdGradE[0] << "\n";
    }
    std::cout << std::endl;
    dataFile.close();
}

// calculate minimum using gradient descent, and save descent steps to a file
template <size_t N, size_t d>
double gradAlphaSampler(
    double alpha0, double learningRate, size_t maxSteps, const MCMode mode, double magnitude,
    SamplerArgs args, size_t mcCycleCount, size_t walkerCount, std::string filename)
{
    if (!std::isnan(args.beta)) {
        assert(d == 3);
    }

    size_t cycleCount = mcCycleCount / std::sqrt(N) / 100;
    double cycleFactor = std::pow(100, 1./(double)maxSteps);

    std::cout << "N=" << N << ", d=" << d << std::endl;
    std::ofstream dataFile(filename);
    if (!dataFile)
    {
        std::cerr << "Error opening " + filename + ": " << strerror(errno) << std::endl;
        exit(0);
    }

    dataFile.precision(14);

    double moment = 0;
    double decay = 0.9;
    
    double minGrad = std::numeric_limits<double>::infinity();
    double bestAlpha = alpha0;

    double grad = std::nan("");
    double alpha = alpha0;
    for (size_t i = 0; i < maxSteps; i++)
    {
        MCSamplerOut out;
        if (std::isnan(args.beta))
        {
            out = mcSampler<N, d>(
                mode, magnitude, cycleCount, cycleCount / 100,
                walkerCount, args.a, args.stateSize, alpha);
        }
        else
        {
            out = mcSampler<N>(
                mode, magnitude, cycleCount, cycleCount / 100,
                walkerCount, args.a, args.stateSize, alpha, args.beta, args.gamma);
        }
        grad = out.gradE[0];

        printf("\r  i=%ld, alpha=%.8f, grad=%.8g                    ", i+1, alpha, grad);
        std::fflush(stdout);

        dataFile << alpha << " " << out.E << " " << out.stdE << "\n";

        if (std::abs(grad) < std::abs(minGrad)) {
            bestAlpha = alpha;
            minGrad = grad;
        }

        if (std::abs(grad) < 1e-14)
        {
            break;
        }

        moment = decay * moment + (1 - decay) * grad * grad;
        alpha = alpha - learningRate * grad / (std::sqrt(moment) + 1e-14);

        cycleCount = (size_t)(cycleCount*cycleFactor);
    }
    printf("\n  Best alpha: %.8f, with grad=%.8g\n", bestAlpha, minGrad);

    dataFile.close();

    return bestAlpha;
}

// use a binary search inspired alogrithm to find the magnitude that minimizes the mean
// relative differences for all the alpha values in alphaVec
template <size_t N, size_t d>
double calibrateMagnitude(
    std::vector<double> &alphaVec, const MCMode mode, size_t mcCycleCount, size_t walkerCount, std::string filename)
{
    size_t cycleCount = mcCycleCount / std::sqrt(N);

    std::ofstream dataFile(filename);
    if (!dataFile)
    {
        std::cerr << "Error opening " + filename + ": " << strerror(errno) << std::endl;
        exit(0);
    }


    dataFile.precision(14);
    std::cout << "N=" << N << ", d=" << d << std::endl;

    // getDiff returns the mean realtive difference, and saves the results to dataFile
    auto getDiff = [&](double magnitude)
    {
        std::cout << "  " << magnitude;
        std::fflush(stdout);

        double avgDiff = 0;
        double avgStd = 0;
        for (auto &alpha : alphaVec)
        {
            MCSamplerOut out = mcSampler<N, d>(
                mode, magnitude, cycleCount, cycleCount / 100, walkerCount, 0, 1, alpha);
            double analVal = (double)(d*N) * 0.5 * (alpha + 1. / (4. * alpha));

            avgDiff += std::abs(out.E - analVal) / analVal;
            avgStd += out.stdE / analVal;
        }
        avgDiff /= (double)alphaVec.size();
        avgStd /= (double)alphaVec.size();

        std::cout << " " << avgDiff << "\n";
        dataFile << magnitude << " " << avgDiff << " " << avgStd << "\n";

        return avgDiff;
    };

    double initialMagnitude = 1;
    double initialDiff = getDiff(initialMagnitude);
    double halfDiff = getDiff(0.5 * initialMagnitude);
    double doubleDiff = getDiff(2.0 * initialMagnitude);

    double diff;
    double left;
    double right;
    bool didDouble;

    if (initialDiff < halfDiff && initialDiff < doubleDiff)
    {
        // neither halving or doubling the magnitude improved results.
        // here we should really change the scale factor from 2 to a smaller
        // value and try again, but I did not think about that - oops
        dataFile.close();
        return initialMagnitude;
    }
    else if (halfDiff < doubleDiff)
    {
        // halving the magnitude improved the result, try halving it more
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

        left = magnitude / 2;
        right = magnitude;
        didDouble = false;
    }
    else
    {
        // doubling the magnitude improved the result, try doubling it more
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
            // test left and right of mid (first and third quarter)
            double q1 = (left + mid) * 0.5;
            double q3 = (mid + right) * 0.5;

            double q1Diff = getDiff(q1);
            double q3Diff = getDiff(q3);

            if (midDiff < q1Diff && midDiff < q3Diff)
            {
                // left and right are worse, we reached a minimum
                dataFile.close();
                return mid;
            }
            else if (q1Diff < q3Diff)
            {
                // right is better
                right = mid;
                mid = q1;
                midDiff = q1Diff;
            }
            else
            {
                // left is better
                left = mid;
                mid = q3;
                midDiff = q3Diff;
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
