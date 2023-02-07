#pragma once

#include <memory>
#include <vector>

#include "utils.hh"
#include "Random.hh"

template <size_t N, size_t d>
ParticleRay<N, d> createUniformState(double a, double boxSize, Random &randomEngine);
