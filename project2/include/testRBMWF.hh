#pragma once

#include "RBMWF.hh"

#define EQUALS(a, b) std::abs(a - b)/std::abs(a) < 1e-14

bool testPerturbation(Random &random)
{
    bool gotErr = false;

    const size_t N = 2;
    const size_t d = 3;
    const size_t M = 5;

    double sigma = 1.0;
    Vec<M> b(arma::fill::ones);
    Vec<N * d> a(arma::fill::randn);
    Mat<N * d, M> W(arma::fill::randn);

    RBMWF<N, d, M> wf = RBMWF<N, d, M>(a, b, W, sigma, MCMode::METHAS);

    ParticleSystem<N, d> state(sigma, random);
    wf.setState(state);
    wf.evaluate();

    wf.pertubateState(0, 1, random);

    // after perturbation
    double pertVal = wf.evaluate();
    QForceMat<N, d> pertQ = wf.computeQForce();

    ParticleSystem<N, d> newState = wf.getState();
    wf.setState(newState);

    // after reset
    double trueVal = wf.evaluate();
    QForceMat<N, d> trueQ = wf.computeQForce();

    RBMWF<N, d, M> newWF = RBMWF<N, d, M>(a, b, W, sigma, MCMode::METHAS);
    newWF.setState(newState);
    wf.setState(state);
    wf.updateFrom(newWF, 0);

    // after update from
    double updVal = wf.evaluate();
    QForceMat<N, d> updQ = wf.computeQForce();

    if (!(EQUALS(trueVal, pertVal) && EQUALS(trueQ[0][1], pertQ[0][1]) && EQUALS(trueQ[1][2], pertQ[1][2])))
    {
        print("Perturbate does not update correctly!");
        fprintf(stderr, "Correct: %.3g, %.3g, %.3g\n", trueVal, trueQ[0][1], trueQ[1][2]);
        fprintf(stderr, "    Got: %.3g, %.3g, %.3g\n", pertVal, pertQ[0][1], pertQ[1][2]);
        fprintf(
            stderr, "   Diff: %.3g, %.3g, %.3g\n",
            std::abs(trueVal - pertVal) / std::abs(trueVal),
            std::abs(trueQ[0][1] - pertQ[0][1]) / std::abs(trueQ[0][1]),
            std::abs(trueQ[1][2] - pertQ[1][2]) / std::abs(trueQ[1][2])
        );
        gotErr = true;
    }

    if (!(EQUALS(trueVal, updVal) && EQUALS(trueQ[0][1], updQ[0][1]) && EQUALS(trueQ[1][2], updQ[1][2])))
    {
        print("Update from does not update correctly!");
        fprintf(stderr, "Correct: %.3g, %.3g, %.3g\n", trueVal, trueQ[0][1], trueQ[1][2]);
        fprintf(stderr, "    Got: %.3g, %.3g, %.3g\n", updVal, updQ[0][1], updQ[1][2]);
        fprintf(
            stderr, "   Diff: %.3g, %.3g, %.3g\n",
            std::abs(trueVal - updVal) / std::abs(trueVal),
            std::abs(trueQ[0][1] - updQ[0][1]) / std::abs(trueQ[0][1]),
            std::abs(trueQ[1][2] - updQ[1][2]) / std::abs(trueQ[1][2]));
        gotErr = true;
    }

    if (!gotErr) {
        print("testPerturbation passed all tests!");
    }

    return gotErr;
}

bool testValues1(Random &random)
{
    bool gotErr = false;

    const size_t N = 2;
    const size_t d = 3;
    const size_t M = 5;

    double sigma = 0.5;
    Vec<M> b(arma::fill::zeros);
    Vec<N * d> a(arma::fill::zeros);
    Mat<N * d, M> W(arma::fill::zeros);

    RBMWF<N, d, M> wf = RBMWF<N, d, M>(a, b, W, sigma, MCMode::METHAS);

    ParticleSystem<N, d> state(sigma, random);
    wf.setState(state);

    double val = wf.evaluate();
    QForceMat<N, d> qForce = wf.computeQForce();
    double localEnergy = wf.computeLocalEnergy();
    RBMGrad grad = wf.computeLogGrad();

    // simple values computed from scratch with a=b=W=0 : note: this is a massive simplification
    double squareSum = state.getSquareSum();
    double isig2 = 1.0 / (sigma * sigma);
    double pow2 = (double)std::pow(2, M);
    double simpleVal = std::exp(-0.5 * isig2 * squareSum) * pow2;
    double simpleE = 0.5 * (double)(N *d) * isig2 + 0.5 * (1 - isig2 * isig2) * squareSum;

    QForceMat<N, d> simpleQ;
    for (size_t i = 0; i < N; i++)
    {
        auto &pos = state[i].getPosition();
        for (size_t j = 0; j < d; j++)
        {
            simpleQ[i][j] = -2.0 * isig2 * pos[j];
        }
    }

    if (!(EQUALS(val, simpleVal))) {
        print("Computation of value is wrong!");
        fprintf(stderr, "%f ≠ %f\n", val, simpleVal);
        gotErr = true;
    }

    if (!(EQUALS(localEnergy, simpleE)))
    {
        print("Computation of local energy is wrong!");
        fprintf(stderr, "%f ≠ %f\n", localEnergy, simpleE);
        gotErr = true;
    }

    if (!(EQUALS(qForce[0][0], simpleQ[0][0]) && EQUALS(qForce[1][2], simpleQ[1][2])))
    {
        print("Computation of qForce is wrong!");
        fprintf(stderr, "(%f, %f) ≠ (%f, %f)\n", qForce[0][0], qForce[1][2], simpleQ[0][0], simpleQ[1][2]);
        gotErr = true;
    }

    if (!gotErr) {
        print("testValues1 passed all tests!");
    }

    return gotErr;
}

bool testValues2(Random &random) {
    bool gotErr = false;

    const size_t N = 1;
    const size_t d = 1;
    const size_t M = 1;

    double sigma = 0.5;
    Vec<M> b(arma::fill::randn);
    Vec<N * d> a(arma::fill::randn);
    Mat<N * d, M> W(arma::fill::randn);

    RBMWF<N, d, M> wf = RBMWF<N, d, M>(a, b, W, sigma, MCMode::METHAS);

    ParticleSystem<N, d> state(sigma, random);
    wf.setState(state);

    double val = wf.evaluate();
    QForceMat<N, d> qForce = wf.computeQForce();
    double localEnergy = wf.computeLocalEnergy();
    RBMGrad grad = wf.computeLogGrad();

    // simple values computed from scratch with N=d=M=0
    double a_val = a(0);
    double b_val = b(0);
    double W_val = W(0, 0);
    double x = state[0].getPosition()[0];

    double isig2 = 1.0 / (sigma * sigma);
    double f = std::exp(b_val + x * W_val * isig2);

    double simpleVal = std::exp(-0.5 * isig2 * (x - a_val) * (x - a_val)) * (1 + f);

    double term1 = W_val * (f / (1 + f)) * (W_val - 2 * (x - a_val));
    double term2 = (x - a_val) * (x - a_val) - sigma * sigma;
    double simpleE = -0.5 * isig2 * isig2 * (term1 + term2) + 0.5 * x * x;  

    double simpleQ = 2.0 * isig2 * (a_val - x + W_val * f / (1 + f));

    double simpleAGrad = isig2 * (x - a_val);
    double simpleBGrad = f / (1 + f);
    double simpleWGrad = x * isig2 * f / (1 + f);

    if (!(EQUALS(val, simpleVal)))
    {
        print("Computation of value is wrong!");
        fprintf(stderr, "%f ≠ %f\n", val, simpleVal);
        gotErr = true;
    }

    if (!(EQUALS(localEnergy, simpleE)))
    {
        print("Computation of local energy is wrong!");
        fprintf(stderr, "%f ≠ %f\n", localEnergy, simpleE);
        gotErr = true;
    }

    if (!(EQUALS(qForce[0][0], simpleQ)))
    {
        print("Computation of qForce is wrong!");
        fprintf(stderr, "%f ≠ %f\n", qForce[0][0], simpleQ);
        gotErr = true;
    }

    if (!(EQUALS(grad.aGrad(0), simpleAGrad)))
    {
        print("Computation of a gradient is wrong!");
        fprintf(stderr, "%f ≠ %f\n", grad.aGrad(0), simpleAGrad);
        gotErr = true;
    }

    if (!(EQUALS(grad.bGrad(0), simpleBGrad)))
    {
        print("Computation of b gradient is wrong!");
        fprintf(stderr, "%f ≠ %f\n", grad.bGrad(0), simpleBGrad);
        gotErr = true;
    }

    if (!(EQUALS(grad.WGrad(0, 0), simpleWGrad)))
    {
        print("Computation of W gradient is wrong!");
        fprintf(stderr, "%f ≠ %f\n", grad.WGrad(0, 0), simpleWGrad);
        gotErr = true;
    }

    if (!gotErr) {
        print("testValues2 passed all tests!");
    }

    return gotErr;
}

void testRBMWF()
{
    // arma::arma_rng::set_seed(123);
    // Random random(123);
    Random random;

    testPerturbation(random);
    testValues1(random);
    testValues2(random);
}
