#pragma once

#include "mcSampler.hh"

#include <string.h>

// calculate minimum using gradient descent, and save descent steps to a file
template <size_t N, size_t d, size_t M>
std::pair<double, RBMParams<N, d, M>> minimizer(
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
    double minE = std::numeric_limits<double>::infinity();

    fprintf(stderr, "N=%ld, d=%ld, M=%ld\n", N, d, M);
    for (size_t i = 0; i < maxSteps; i++)
    {
        MCSamplerOut<N, d, M> out = mcSampler<N, d, M>(
            mode, magnitude, cycleCount, cycleCount / 100, walkerCount, sigma, interactions, params);
        fprintf(stderr, "\r  i=%ld, E=%.8f            ", i+1, out.E);
        if (filename != "") {
            dataFile << i << " " << out.E << " " << out.stdE << "\n";
        }

        if (i > (maxSteps - 10) && out.E < minE)
        {
            minE = out.E;
            bestParams = params;
        }

        moment.aGrad = decay * moment.aGrad + (1 - decay) * out.gradE.aGrad % out.gradE.aGrad;
        moment.bGrad = decay * moment.bGrad + (1 - decay) * out.gradE.bGrad % out.gradE.bGrad;
        moment.WGrad = decay * moment.WGrad + (1 - decay) * out.gradE.WGrad % out.gradE.WGrad;

        params.a = params.a - learningRate * out.gradE.aGrad / (arma::sqrt(moment.aGrad) + 1e-14);
        params.b = params.b - learningRate * out.gradE.bGrad / (arma::sqrt(moment.bGrad) + 1e-14);
        params.W = params.W - learningRate * out.gradE.WGrad / (arma::sqrt(moment.WGrad) + 1e-14);

        cycleCount = (size_t)(cycleCount * cycleFactor);
    }
    printf("\n  Minimum energy: %.8f\n", minE);

    if (filename != "") {
        dataFile.close();
    }

    return {minE, bestParams};
}
