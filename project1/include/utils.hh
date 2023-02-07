#pragma once

#include <array>

#include "particle.hh"

template <size_t N, size_t d>
using ParticleRay = std::array<std::unique_ptr<Particle<d>>, N>;
