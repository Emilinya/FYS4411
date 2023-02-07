#include <memory>
#include <assert.h>

#include "include/createState.hh"

template <size_t N, size_t d>
ParticleRay<N, d> createUniformState(double a, double boxSize, Random &random)
{
    assert(N > 0 && d > 0);

    ParticleRay<N, d> particles;

    for (size_t i = 0; i < N; i++)
    {
        std::array<double, d> position;

        for (size_t j = 0; j < d; j++)
        {
            position[j] = random.nextDouble(0, boxSize);
        }

        particles[i] = std::make_unique<Particle>(a, position);
    }

    return particles;
}
