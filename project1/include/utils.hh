#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <cmath>

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

// This function acts like np.linspace
std::vector<double> linspace(double a, double b, size_t n)
{
    std::vector<double> vec;
    vec.reserve(n);
    for (size_t i = 0; i < n; i++)
    {
        double p = (double)i / (double)(n - 1);
        vec.push_back(a + p * (b - a));
    }
    return vec;
}

// function to calculate mean and std of a vector
std::tuple<double, double> calcMeanStd(std::vector<double> &vals)
{
    double mean = 0;
    for (size_t i = 0; i < vals.size(); i++)
    {
        mean += vals[i];
    }
    mean /= (double)vals.size();

    double std = 0;
    for (size_t i = 0; i < vals.size(); i++)
    {
        double diff = (vals[i]  - mean);
        std += diff * diff;
    }
    std = std::sqrt(std / (double)vals.size());

    return {mean, std};
}

// the following functions define a print function that acts like python's print

inline void print()
{
    std::cerr << '\n';
}

template <typename T>
inline void print(T arg)
{
    std::cerr << arg << '\n';
}

template <typename T, typename... Rest>
inline void print(T arg, Rest... rest)
{
    std::cerr << arg << ' ';
    print(rest...);
}

#define rprint(stuff) std::cerr << "\r" << stuff << "          "
