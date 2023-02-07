#include <memory>

#include "include/waveFunction.hh"
#include "include/createState.hh"
#include "include/particle.hh"
#include "include/Random.hh"
#include "include/utils.hh"

int main()
{
    // Metropolis parameters
    unsigned int stepCount = 1e6;
    unsigned int equilibrationSteps = 1e5;
    double stepLength = 0.1;

    const size_t d = 1; // dimensions
    const size_t N = 1; // number of particles
    double omega = 1.0;
    double alpha = 0.5;

    Random rng;
    ParticleRay<N, d> particles = createUniformState<N, d>(0, 1, rng);

    return 0;
}
