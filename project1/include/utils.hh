#pragma once

#include <array>

#include "Particle.hh"

template <size_t N, size_t d>
using ParticleRay = std::array<Particle<d>, N>;

template <size_t N, size_t d>
using QForceMat = std::array<std::array<double, d>, N>;

template <size_t d>
void printRay(std::array<double, d> &array) {
    if (d==0) {
        std::cout << "()\n";
    }

    std::cout << "(";
    for (size_t i = 0; i < d-1; i++)
    {
        std::cout << array[i] << ", ";
    }
    std::cout << array[d-1] << ")\n";
    

}
