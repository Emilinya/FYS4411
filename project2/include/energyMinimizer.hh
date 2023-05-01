#pragma once

#include "mcSampler.hh"

#include <string.h>

// calculate minimum using gradient descent, and save descent steps to a file
template <size_t N, size_t d, size_t M>
RBMParams<N, d, M> minimizer(
    double learningRate, size_t maxSteps, const MCMode mode, double magnitude,
    double sigma, bool interactions, size_t mcCycleCount, size_t walkerCount, std::string filename = "")
{
    size_t cycleCount = mcCycleCount / 100;
    double cycleFactor = std::pow(100, 1. / (double)maxSteps);

    std::ofstream dataFile;
    if (filename != "") {
        dataFile = errcheckOpen(filename).value();
        dataFile.precision(14);
    }

    RBMGrad<N, d, M> moment;
    double decay = 0.9;

    RBMParams<N, d, M> params;
    RBMParams<N, d, M> bestParams;
    double minGrad = std::numeric_limits<double>::infinity();

    fprintf(stderr, "N=%ld, d=%ld, M=%ld\n", N, d, M);
    for (size_t i = 0; i < maxSteps; i++)
    {
        MCSamplerOut<N, d, M> out = mcSampler<N, d, M>(
            mode, magnitude, cycleCount, cycleCount / 100, walkerCount, sigma, interactions, params);
        if (filename != "") {
            dataFile << i << " " << out.E << " " << out.stdE << "\n";
        }

        auto aGrad2 = out.gradE.aGrad % out.gradE.aGrad;
        auto bGrad2 = out.gradE.bGrad % out.gradE.bGrad;
        auto WGrad2 = out.gradE.WGrad % out.gradE.WGrad;

        double gradSize = arma:: accu(aGrad2) + arma:: accu(bGrad2) + arma:: accu(WGrad2);

        fprintf(stderr, "\r  i=%ld, E=%.8f, grad=%.8f            ", i + 1, out.E, gradSize);

        if (gradSize < minGrad)
        {
            minGrad = gradSize;
            bestParams = params;
        }

        moment.aGrad = decay * moment.aGrad + (1 - decay) * aGrad2;
        moment.bGrad = decay * moment.bGrad + (1 - decay) * bGrad2;
        moment.WGrad = decay * moment.WGrad + (1 - decay) * WGrad2;

        params.a = params.a - learningRate * out.gradE.aGrad / (arma::sqrt(moment.aGrad) + 1e-14);
        params.b = params.b - learningRate * out.gradE.bGrad / (arma::sqrt(moment.bGrad) + 1e-14);
        params.W = params.W - learningRate * out.gradE.WGrad / (arma::sqrt(moment.WGrad) + 1e-14);

        cycleCount = (size_t)(cycleCount * cycleFactor);
    }
    printf("\n  Minimum gradient: %.8f\n", minGrad);

    if (filename != "") {
        dataFile.close();
    }

    return bestParams;
}
