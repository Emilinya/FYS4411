#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <cmath>
#include <iomanip>

#include <armadillo>

template <size_t N, size_t d>
using QForceMat = std::array<std::array<double, d>, N>;

template <uint N>
using Vec = arma::dvec::fixed<N>;

typedef Vec<3> Vec3;

enum class MCMode
{
    MET,
    METHAS,
};

struct MCStepOut
{
    double E;
    std::vector<double> logGrad;
};

struct MCSamplerOut
{
    double E;
    double stdE;
    std::vector<double> gradE;
    std::vector<double> stdGradE;
};

template <size_t d>
void printRay(std::array<double, d> &array)
{
    if (d == 0)
    {
        std::cout << "()\n";
    }

    std::cout << "(";
    for (size_t i = 0; i < d - 1; i++)
    {
        std::cout << array[i] << ", ";
    }
    std::cout << array[d - 1] << ")\n";
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
        double diff = (vals[i] - mean);
        std += diff * diff;
    }
    std = std::sqrt(std / (double)vals.size());

    return {mean, std};
}

// function to calculate mean and std of a vector of vectors
std::tuple<std::vector<double>, std::vector<double>>
vectorCalcMeanStd(const std::vector<std::vector<double>> &vals)
{
    size_t outerN = vals.size();
    size_t innerN = vals[0].size();

    std::vector<double> means;
    means.resize(innerN);

    std::vector<double> stds;
    stds.resize(innerN);

    for (size_t i = 0; i < innerN; i++)
    {
        means[i] = 0;
        stds[i] = 0;
    }

    for (size_t i = 0; i < outerN; i++)
    {
        for (size_t j = 0; j < innerN; j++)
        {
            means[j] += vals[i][j];
        }
    }

    for (size_t j = 0; j < innerN; j++)
    {
        means[j] = means[j] / outerN;
    }

    for (size_t i = 0; i < outerN; i++)
    {
        for (size_t j = 0; j < innerN; j++)
        {
            double diff = (vals[i][j] - means[j]);
            stds[j] += diff * diff;
        }
    }

    for (size_t i = 0; i < innerN; i++)
    {
        stds[i] = std::sqrt(stds[i] / (double)outerN);
    }

    return {means, stds};
}

// the following functions define a print function that acts like python's print

inline void print()
{
    std::cerr << '\n';
}

template <typename T>
inline void print(T arg)
{
    std::cerr << std::setprecision(std::numeric_limits<T>::digits10) << arg << '\n';
}

template <typename T, typename... Rest>
inline void print(T arg, Rest... rest)
{
    std::cerr << arg << ' ';
    print(rest...);
}

#define rprint(stuff) std::cerr << "\r" << stuff << "          "

// string formatting from https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);
}

std::string prettify_ms(double ms)
{
    if (ms > 1000.)
    {
        int s = static_cast<int>(ms / 1000.);
        if (s > 60.)
        {
            // TODO: Implement this properly
            int m = static_cast<int>(s / 60.);
            ms -= 1000. * s;
            s -= m * 60;
            return string_format("%d m %d s %.0f ms", m, s, ms);
        }
        else
        {
            ms -= 1000. * s;
            if (ms == 0)
            {
                return string_format("%d s", s);
            }
            else
            {
                return string_format("%d s %.0f ms", s, ms);
            }
        }
    }
    else
    {
        return string_format("%.3g ms", ms);
    }
}

class Timer
{
    // This is a timer class to simplify timing (especially removing all those long type declarations)
public:
    Timer();
    double get_ms();
    std::string get_pretty();
    void restart();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

Timer::Timer()
{
    start = std::chrono::high_resolution_clock::now();
}

double Timer::get_ms()
{
    // Get elapsed time in ms as a double
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count();
}

std::string Timer::get_pretty()
{
    // Get elapsed time as a nice, formatted string
    return prettify_ms(Timer::get_ms());
}

void Timer::restart()
{
    start = std::chrono::high_resolution_clock::now();
}
