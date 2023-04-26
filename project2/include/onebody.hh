#include "mcSampler.hh"
#include "omp.h"

// function to calculate the one-body densities of the elliptical wave function
template <size_t N>
void onebodyCalculator(
    const MCMode mode, double magnitude, size_t mcCycleCount, size_t burnCycleCount, size_t walkerCount,
    double diameter, double stateSize, double alpha, double beta, double gamma, std::string filename)
{
    double xsize = 6;
    size_t nbins = 200;
    double binDelta = (2 * xsize) / (double)nbins;

    std::vector<std::vector<size_t>> xcounts(walkerCount, std::vector<size_t>(nbins, 0));

    #pragma omp parallel for
    for (size_t i = 0; i < walkerCount; i++)
    {
        int thread_id = omp_get_thread_num();
        Random random;

        ElipticalWF<N> waveFunction(alpha, beta, gamma, mode);
        while (true)
        {
            // generated state might be impossible. In that case, try again.
            ParticleSystem<N, 3> state(diameter, stateSize);
            if (waveFunction.setState(state))
            {
                break;
            };
        }

        // evaluate once so value is copied to the wave function copy
        waveFunction.evaluate();

        // burn some samples
        for (size_t mcCycle = 0; mcCycle < burnCycleCount; mcCycle++)
        {
            mcStep<N, 3, ElipticalWF<N>>(magnitude, waveFunction, mode, random);
        }

        for (size_t mcCycle = 0; mcCycle < mcCycleCount; mcCycle++)
        {
            // only one thread should print
            if (thread_id == 0) {
                size_t next_pro = 1000 * (mcCycle + 1) / mcCycleCount;
                size_t curr_pro = 1000 * mcCycle / mcCycleCount;
                if (next_pro != curr_pro) {
                    rprint(string_format("%.1f/100", (double)curr_pro/10.));
                }
            }
            mcStep<N, 3, ElipticalWF<N>>(magnitude, waveFunction, mode, random);

            ParticleSystem<N, 3> &state = waveFunction.getState();
            for (size_t j = 0; j < N; j++)
            {
                double x = state[j].getPosition()[0];
                if (std::abs(x) > xsize)
                {
                    print(x, xsize);
                    throw std::runtime_error("onebodyCalculator: xsize is too low!");
                }

                size_t binidx = std::round((x + xsize) / binDelta);
                xcounts[i][binidx] += 1;
            }
        }
    }
    print("\r100/100    ");

    // combine x counts from all walkers
    std::vector<size_t> combiXCounts(nbins, 0);
    for (size_t j = 0; j < walkerCount; j++)
    {
        for (size_t i = 0; i < nbins; i++)
        {
            combiXCounts[i] += xcounts[j][i];
        }
    }

    // calculate the uncertanties (this is unneccesary, but oh well)
    std::vector<double> xcountUncertanties(nbins, 0);
    for (size_t i = 0; i < nbins; i++)
    {
        for (size_t j = 0; j < walkerCount; j++)
        {
            double diff = (double)combiXCounts[i] / (double)walkerCount - (double)xcounts[j][i];
            xcountUncertanties[i] += diff * diff;
        }
        xcountUncertanties[i] = std::sqrt(xcountUncertanties[i] / (double)walkerCount);
    }

    // we want the integral of the onebody densities to be N
    size_t binSum = 0;
    for (auto count : combiXCounts)
    {
        binSum += count * binDelta;
    }
    double normFact = (double)N / (double)binSum;

    std::ofstream dataFile(filename);
    if (!dataFile)
    {
        std::cerr << "Error opening " + filename + ": " << strerror(errno) << std::endl;
        exit(0);
    }

    dataFile.precision(14);
    for (size_t i = 0; i < nbins; i++)
    {
        double x = i * binDelta - xsize;
        double rho = combiXCounts[i] * normFact;
        double std = xcountUncertanties[i] * normFact;
        dataFile << x << " " << rho << " " << std << "\n";
    }
    dataFile.close();
}
