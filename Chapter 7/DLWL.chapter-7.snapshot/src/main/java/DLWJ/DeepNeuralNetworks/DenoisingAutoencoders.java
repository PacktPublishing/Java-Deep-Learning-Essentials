package DLWJ.DeepNeuralNetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import static DLWJ.util.ActivationFunction.sigmoid;
import static DLWJ.util.RandomGenerator.*;


public class DenoisingAutoencoders {

    public int nVisible;
    public int nHidden;
    public double[][] W;
    public double[] hbias;
    public double[] vbias;
    public Random rng;


    public DenoisingAutoencoders(int nVisible, int nHidden, double[][] W, double[] hbias, double[] vbias, Random rng) {

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

    public void train(double[][] X, int minibatchSize, double learningRate, double corruptionLevel) {

        double[][] grad_W = new double[nHidden][nVisible];
        double[] grad_hbias = new double[nHidden];
        double[] grad_vbias = new double[nVisible];

        // train with minibatches
        for (int n = 0; n < minibatchSize; n++) {

            // add noise to original inputs
            double[] corruptedInput = getCorruptedInput(X[n], corruptionLevel);

            // encode
            double[] z = getHiddenValues(corruptedInput);

            // decode
            double[] y = getReconstructedInput(z);


            // calculate gradients

            // vbias
            double[] v_ = new double[nVisible];

            for (int i = 0; i < nVisible; i++) {
                v_[i] = X[n][i] - y[i];
                grad_vbias[i] += v_[i];
            }

            // hbias
            double[] h_ = new double[nHidden];

            for (int j = 0; j < nHidden; j++) {

                for (int i = 0; i < nVisible; i++) {
                    h_[j] = W[j][i] * (X[n][i] - y[i]);
                }

                h_[j] *= z[j] * (1 - z[j]);
                grad_hbias[j] += h_[j];
            }

            // W
            for (int j = 0; j < nHidden; j++) {
                for (int i = 0; i < nVisible; i++) {
                    grad_W[j][i] += h_[j] * corruptedInput[i] + v_[i] * z[j];
                }
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

    public double[] getCorruptedInput(double[] x, double corruptionLevel) {

        double[] corruptedInput = new double[x.length];

        // add masking noise
        for (int i = 0; i < x.length; i++) {
            double rand_ = rng.nextDouble();

            if (rand_ < corruptionLevel) {
                corruptedInput[i] = 0.;
            } else {
                corruptedInput[i] = x[i];
            }
        }

        return corruptedInput;
    }

    public double[] getHiddenValues(double[] x) {

        double[] z = new double[nHidden];

        for (int j = 0; j < nHidden; j++) {
            for (int i = 0; i < nVisible; i++) {
                z[j] += W[j][i] * x[i];
            }

            z[j] += hbias[j];
            z[j] = sigmoid(z[j]);
        }

        return z;
    }

    public double[] getReconstructedInput(double[] z) {

        double[] y = new double[nVisible];

        for (int i = 0; i < nVisible; i++) {
            for (int j = 0; j < nHidden; j++) {
                y[i] += W[j][i] * z[j];
            }

            y[i] += vbias[i];
            y[i] = sigmoid(y[i]);
        }

        return y;
    }

    public double[] reconstruct(double[] x) {

        double[] z = getHiddenValues(x);
        double[] y = getReconstructedInput(z);

        return y;
    }


    public static void main(String[] args) {

        final Random rng = new Random(1234);

        //
        // Declare variables and constants
        //

        int train_N_each = 200;           // for demo
        int test_N_each = 2;              // for demo
        int nVisible_each = 4;           // for demo
        double pNoise_Training = 0.05;     // for demo
        double pNoise_Test = 0.25;         // for demo

        final int patterns = 3;

        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;

        final int nVisible = nVisible_each * patterns;
        int nHidden = 6;
        double corruptionLevel = 0.3;

        double[][] train_X = new double[train_N][nVisible];
        double[][] test_X = new double[test_N][nVisible];
        double[][] reconstructed_X = new double[test_N][nVisible];

        int epochs = 1000;
        double learningRate = 0.2;
        int minibatchSize = 10;
        final int minibatch_N = train_N / minibatchSize;

        double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nVisible];
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
                        train_X[n_][i] = binomial(1, 1 - pNoise_Training, rng);
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
                        test_X[n_][i] = binomial(1, 1 - pNoise_Test, rng);
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
        // Build Denoising Autoencoders Model
        //

        // construct DA
        DenoisingAutoencoders nn = new DenoisingAutoencoders(nVisible, nHidden, null, null, null, rng);

        // train
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatch_N; batch++) {
                nn.train(train_X_minibatch[batch], minibatchSize, learningRate, corruptionLevel);
            }
        }

        // test (reconstruct noised data)
        for (int i = 0; i < test_N; i++) {
            reconstructed_X[i] = nn.reconstruct(test_X[i]);
        }

        // evaluation
        System.out.println("-----------------------------------");
        System.out.println("DA model reconstruction evaluation");
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
