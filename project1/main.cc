#include <memory>
#include <iostream>

#include "WaveFunction.hh"
#include "SphericalWF.hh"
#include "createState.hh"
#include "mcSampler.hh"
#include "Particle.hh"
#include "Random.hh"
#include "utils.hh"

int main()
{
    const size_t d = 1; // dimensions
    const size_t N = 5; // number of particles

    // wave function parameters
    double omega = 1.0;
    double alpha = 0.5;
    SphericalWF<N, d> waveFunction(alpha);

    double diameter = 1;
    double stateSize = 1;

    Random rng;
    ParticleRay<N, d> initialState = createUniformState<N, d>(diameter, stateSize, rng);

    // Fokker-Planck parameters
    double D = 0.5;
    double timeStep = 0.05;

    // MCMC parameters
    unsigned int mcCycleCount = 10;

    auto [E, E2] = monteCarloSampler<N, d>(D, timeStep, mcCycleCount, initialState, waveFunction, rng);
    std::cout << E << " " << E2 << "\n";

    return 0;
}
