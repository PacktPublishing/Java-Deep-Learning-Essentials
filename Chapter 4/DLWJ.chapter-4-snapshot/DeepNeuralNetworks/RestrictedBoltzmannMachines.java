package DLWJ.DeepNeuralNetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import static DLWJ.util.ActivationFunction.sigmoid;
import static DLWJ.util.RandomGenerator.*;


public class RestrictedBoltzmannMachines {

    public int nVisible;
    public int nHidden;
    public double[][] W;
    public double[] hbias;
    public double[] vbias;
    public Random rng;

    public RestrictedBoltzmannMachines(int nVisible, int nHidden, double[][] W, double[] hbias, double[] vbias, Random rng) {

        if (rng == null) rng = new Random(1234);  // seed random

        if (W == null) {

            W = new double[nHidden][nVisible];
            double w_ = 1. / nVisible;

            for (int j = 0; j < nHidden; j++) {
                for (int i = 0; i < nVisible; i++) {
                    W[j][i] = uniform(-w_, w_, rng);
                }
            }
        }

        if (hbias == null) {
            hbias = new double[nHidden];

            for (int j = 0; j < nHidden; j++) {
                hbias[j] = 0.;
            }
        }

        if (vbias == null) {
            vbias = new double[nVisible];

            for (int i = 0; i < nVisible; i++) {
                vbias[i] = 0.;
            }
        }

        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.W = W;
        this.hbias = hbias;
        this.vbias = vbias;
        this.rng = rng;

    }

    public void contrastiveDivergence(int[][] X, int minibatchSize, double learningRate, int k) {

        double[][] grad_W = new double[nHidden][nVisible];
        double[] grad_hbias = new double[nHidden];
        double[] grad_vbias = new double[nVisible];

        // train with minibatches
        for (int n = 0; n < minibatchSize; n++) {

            double[] phMean_ = new double[nHidden];
            int[] phSample_ = new int[nHidden];
            double[] nvMeans_ = new double[nVisible];
            int[] nvSamples_ = new int[nVisible];
            double[] nhMeans_ = new double[nHidden];
            int[] nhSamples_ = new int[nHidden];

            // CD-k : CD-1 is enough for sampling (i.e. k = 1)
            sampleHgivenV(X[n], phMean_, phSample_);

            for (int step = 0; step < k; step++) {

                // Gibbs sampling
                if (step == 0) {
                    gibbsHVH(phSample_, nvMeans_, nvSamples_, nhMeans_, nhSamples_);
                } else {
                    gibbsHVH(nhSamples_, nvMeans_, nvSamples_, nhMeans_, nhSamples_);
                }

            }

            // calculate gradients
            for (int j = 0; j < nHidden; j++) {
                for (int i = 0; i < nVisible; i++) {
                    grad_W[j][i] += phMean_[j] * X[n][i] - nhMeans_[j] * nvSamples_[i];
                }

                grad_hbias[j] += phMean_[j] - nhMeans_[j];
            }

            for (int i = 0; i < nVisible; i++) {
                grad_vbias[i] += X[n][i] - nvSamples_[i];
            }

        }

        // update params
        for (int j = 0; j < nHidden; j++) {
            for (int i = 0; i < nVisible; i++) {
                W[j][i] += learningRate * grad_W[j][i] / minibatchSize;
            }

            hbias[j] += learningRate * grad_hbias[j] / minibatchSize;
        }

        for (int i = 0; i < nVisible; i++) {
            vbias[i] += learningRate * grad_vbias[i] / minibatchSize;
        }

    }

    public void gibbsHVH(int[] h0Sample, double[] nvMeans, int[] nvSamples, double[] nhMeans, int[] nhSamples) {
        sampleVgivenH(h0Sample, nvMeans, nvSamples);
        sampleHgivenV(nvSamples, nhMeans, nhSamples);
    }

    public void sampleHgivenV(int[] v0Sample, double[] mean, int[] sample) {

        for (int j = 0; j < nHidden; j++) {
            mean[j] = propup(v0Sample, W[j], hbias[j]);
            sample[j] = binomial(1, mean[j], rng);
        }

    }

    public void sampleVgivenH(int[] h0Sample, double[] mean, int[] sample) {

        for(int i = 0; i < nVisible; i++) {
            mean[i] = propdown(h0Sample, i, vbias[i]);
            sample[i] = binomial(1, mean[i], rng);
        }
    }

    public double propup(int[] v, double[] w, double bias) {

        double preActivation = 0.;

        for (int i = 0; i < nVisible; i++) {
            preActivation += w[i] * v[i];
        }
        preActivation += bias;

        return sigmoid(preActivation);
    }

    public double propdown(int[] h, int i, double bias) {

        double preActivation = 0.;

        for (int j = 0; j < nHidden; j++) {
            preActivation += W[j][i] * h[j];
        }
        preActivation += bias;

        return sigmoid(preActivation);
    }

    public double[] reconstruct(int[] v) {

        double[] x = new double[nVisible];
        double[] h = new double[nHidden];

        for (int j = 0; j < nHidden; j++) {
            h[j] = propup(v, W[j], hbias[j]);
        }

        for (int i = 0; i < nVisible; i++) {
            double preActivation_ = 0.;

            for (int j = 0; j < nHidden; j++) {
                preActivation_ += W[j][i] * h[j];
            }
            preActivation_ += vbias[i];

            x[i] = sigmoid(preActivation_);
        }

        return x;
    }



    public static void main(String[] args) {

        final Random rng = new Random(123);

        //
        // Declare variables and constants
        //

        int train_N_each = 200;         // for demo
        int test_N_each = 2;            // for demo
        int nVisible_each = 4;          // for demo
        double pNoise_Training = 0.05;  // for demo
        double pNoise_Test = 0.25;      // for demo

        final int patterns = 3;

        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;

        final int nVisible = nVisible_each * patterns;
        int nHidden = 6;

        int[][] train_X = new int[train_N][nVisible];
        int[][] test_X = new int[test_N][nVisible];
        double[][] reconstructed_X = new double[test_N][nVisible];

        int epochs = 1000;
        double learningRate = 0.2;
        int minibatchSize = 10;
        final int minibatch_N = train_N / minibatchSize;

        int[][][] train_X_minibatch = new int[minibatch_N][minibatchSize][nVisible];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);


        //
        // Create training data and test data for demo.
        //   Data without noise would be:
        //     class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        //     class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        //     class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        //   and to each data, we add some noise.
        //   For example, one of the data in class 1 could be:
        //     [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        //

        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < train_N_each; n++) {

                int n_ = pattern * train_N_each + n;

                for (int i = 0; i < nVisible; i++) {
                    if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
                            (i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
                        train_X[n_][i] = binomial(1, 1-pNoise_Training, rng);
                    } else {
                        train_X[n_][i] = binomial(1, pNoise_Training, rng);
                    }
                }
            }

            for (int n = 0; n < test_N_each; n++) {

                int n_ = pattern * test_N_each + n;

                for (int i = 0; i < nVisible; i++) {
                    if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
                            (i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
                        test_X[n_][i] = binomial(1, 1-pNoise_Test, rng);
                    } else {
                        test_X[n_][i] = binomial(1, pNoise_Test, rng);
                    }
                }
            }
        }


        // create minibatches
        for (int i = 0; i < minibatch_N; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
            }
        }


        //
        // Build Restricted Boltzmann Machine Model
        //

        // construct RBM
        RestrictedBoltzmannMachines nn = new RestrictedBoltzmannMachines(nVisible, nHidden, null, null, null, rng);

        // train with contrastive divergence
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatch_N; batch++) {
                nn.contrastiveDivergence(train_X_minibatch[batch], minibatchSize, learningRate, 1);
            }
            learningRate *= 0.995;
        }

        // test (reconstruct noised data)
        for (int i = 0; i < test_N; i++) {
            reconstructed_X[i] = nn.reconstruct(test_X[i]);
        }

        // evaluation
        System.out.println("-----------------------------------");
        System.out.println("RBM model reconstruction evaluation");
        System.out.println("-----------------------------------");

        for (int pattern = 0; pattern < patterns; pattern++) {

            System.out.printf("Class%d\n", pattern + 1);

            for (int n = 0; n < test_N_each; n++) {

                int n_ = pattern * test_N_each + n;

                System.out.print( Arrays.toString(test_X[n_]) + " -> ");
                System.out.print("[");
                for (int i = 0; i < nVisible-1; i++) {
                    System.out.printf("%.5f, ", reconstructed_X[n_][i]);
                }
                System.out.printf("%.5f]\n", reconstructed_X[n_][nVisible-1]);
            }

            System.out.println();
        }

    }
}
