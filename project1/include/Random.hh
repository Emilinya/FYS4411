#pragma once

#include <random>

class Random
{
public:
    Random()
    {
        std::random_device rd;
        engine_ = std::mt19937_64(rd());
    }

    Random(int seed)
    {
        engine_ = std::mt19937_64(seed);
    }

    // Uniformly distributed random integers from [min, max].
    int nextInt(int min, int max)
    {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(engine_);
    }

    // Uniformly distributed random doubles from [min, max).
    double nextDouble(double min, double max)
    {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(engine_);
    }

    // Uniformly distributed random doubles from N(mean, std).
    double nextGaussian(double mean, double std)
    {
        std::normal_distribution<double> dist(mean, std);
        return dist(engine_);
    }

private:
    std::mt19937_64 engine_;
};
