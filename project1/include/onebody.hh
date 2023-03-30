#include "mcSampler.hh"

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

    std::vector<size_t> combiXCounts(nbins, 0);
    for (size_t j = 0; j < walkerCount; j++)
    {
        for (size_t i = 0; i < nbins; i++)
        {
            combiXCounts[i] += xcounts[j][i];
        }
    }
    

    // we want the integral of the onebody densities to be N
    size_t binSum = 0;
    for (auto count : combiXCounts)
    {
        binSum += count;
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
        dataFile << x << " " << rho << "\n";
    }
    dataFile.close();
}
