package DLWJ.DeepNeuralNetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import DLWJ.MultiLayerNeuralNetworks.HiddenLayer;
import DLWJ.SingleLayerNeuralNetworks.LogisticRegression;
import static DLWJ.util.RandomGenerator.binomial;


public class Dropout {

    public int nIn;
    public int[] hiddenLayerSizes;
    public int nOut;
    public int nLayers;
    public HiddenLayer[] hiddenLayers;
    public LogisticRegression logisticLayer;
    public Random rng;

    public Dropout(int nIn, int[] hiddenLayerSizes, int nOut, Random rng, String activation) {

        if (rng == null) rng = new Random(1234);

        if (activation == null) activation = "ReLU";

        this.nIn = nIn;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.nOut = nOut;
        this.nLayers = hiddenLayerSizes.length;
        this.hiddenLayers = new HiddenLayer[nLayers];
        this.rng = rng;

        // construct multi-layer
        for (int i = 0; i < nLayers; i++) {
            int nIn_;
            if (i == 0) nIn_ = nIn;
            else nIn_ = hiddenLayerSizes[i - 1];

            // construct hidden layer
            hiddenLayers[i] = new HiddenLayer(nIn_, hiddenLayerSizes[i], null, null, rng, activation);
        }

        // construct logistic layer
        logisticLayer = new LogisticRegression(hiddenLayerSizes[nLayers - 1], nOut);
    }

    public void train(double[][] X, int[][] T, int minibatchSize, double learningRate, double pDrouput) {

        List<double[][]> layerInputs = new ArrayList<>(nLayers+1);
        layerInputs.add(X);

        List<int[][]> dropoutMasks = new ArrayList<>(nLayers);

        double[][] Z = new double[0][0];
        double[][] D; // delta

        // forward hidden layers
        for (int layer = 0; layer < nLayers; layer++) {

            double[] x_;  // layer input
            double[][] Z_ = new double[minibatchSize][hiddenLayerSizes[layer]];
            int[][] mask_ = new int[minibatchSize][hiddenLayerSizes[layer]];

            for (int n = 0; n < minibatchSize; n++) {

                if (layer == 0) {
                    x_ = X[n];
                } else {
                    x_ = Z[n];
                }


                Z_[n] = hiddenLayers[layer].forward(x_);
                mask_[n] = dropout(Z_[n], pDrouput);  // apply dropout mask to units
            }

            Z = Z_;
            layerInputs.add(Z.clone());

            dropoutMasks.add(mask_);
        }

        // forward & backward output layer
        D = logisticLayer.train(Z, T, minibatchSize, learningRate);

        // backward hidden layers
        for (int layer = nLayers - 1; layer >= 0; layer--) {

            double[][] Wprev_;

            if (layer == nLayers - 1) {
                Wprev_ = logisticLayer.W;
            } else {
                Wprev_ = hiddenLayers[layer+1].W;
            }

            // apply mask to delta as well
            for (int n = 0; n < minibatchSize; n++) {
                int[] mask_ = dropoutMasks.get(layer)[n];

                for (int j = 0; j < D[n].length; j++) {
                    D[n][j] *= mask_[j];
                }
            }

            D = hiddenLayers[layer].backward(layerInputs.get(layer), layerInputs.get(layer+1), D, Wprev_, minibatchSize, learningRate);
        }
    }

    public int[] dropout(double[] z, double p) {

        int size = z.length;
        int[] mask = new int[size];

        for (int i = 0; i < size; i++) {
            mask[i] = binomial(1, 1 - p, rng);
            z[i] *= mask[i]; // apply mask
        }

        return mask;
    }

    public void pretest(double pDropout) {

        for (int layer = 0; layer < nLayers; layer++) {

            int nIn_, nOut_;

            if (layer == 0) {
                nIn_ = nIn;
            } else {
                nIn_ = hiddenLayerSizes[layer];
            }

            if (layer == nLayers - 1) {
                nOut_ = nOut;
            } else {
                nOut_ = hiddenLayerSizes[layer+1];
            }

            for (int j = 0; j < nOut_; j++) {
                for (int i = 0; i < nIn_; i++) {
                    hiddenLayers[layer].W[j][i] *= 1 - pDropout;
                }
            }
        }
    }

    public Integer[] predict(double[] x) {

        double[] z = new double[0];

        for (int layer = 0; layer < nLayers; layer++) {

            double[] x_;

            if (layer == 0) {
                x_ = x;
            } else {
                x_ = z.clone();
            }

            z = hiddenLayers[layer].forward(x_);
        }

        return logisticLayer.predict(z);
    }


    public static void main(String[] args) {

        final Random rng = new Random(123);

        //
        // Declare variables and constants
        //

        int train_N_each = 300;        // for demo
        int test_N_each = 50;          // for demo
        int nIn_each = 20;             // for demo
        double pNoise_Training = 0.2;  // for demo
        double pNoise_Test = 0.25;     // for demo

        final int patterns = 3;

        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;

        final int nIn = nIn_each * patterns;
        final int nOut = patterns;


        int[] hiddenLayerSizes = {100, 80};
        double pDropout = 0.5;

        double[][] train_X = new double[train_N][nIn];
        int[][] train_T = new int[train_N][nOut];

        double[][] test_X = new double[test_N][nIn];
        Integer[][] test_T = new Integer[test_N][nOut];
        Integer[][] predicted_T = new Integer[test_N][nOut];

        int epochs = 5000;
        double learningRate = 0.2;

        int minibatchSize = 50;
        final int minibatch_N = train_N / minibatchSize;

        double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];
        int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);


        //
        // Create training data and test data for demo.
        //
        for (int pattern = 0; pattern < patterns; pattern++) {

            for (int n = 0; n < train_N_each; n++) {

                int n_ = pattern * train_N_each + n;

                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
                            (i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
                        train_X[n_][i] = binomial(1, 1 - pNoise_Training, rng) * rng.nextDouble() * .5 + .5;
                    } else {
                        train_X[n_][i] = binomial(1, pNoise_Training, rng) * rng.nextDouble() * .5 + .5;
                    }
                }

                for (int i = 0; i < nOut; i++) {
                    if (i == pattern) {
                        train_T[n_][i] = 1;
                    } else {
                        train_T[n_][i] = 0;
                    }
                }
            }


            for (int n = 0; n < test_N_each; n++) {

                int n_ = pattern * test_N_each + n;

                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
                            (i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
                        test_X[n_][i] = (double) binomial(1, 1 - pNoise_Test, rng) * rng.nextDouble() * .5 + .5;
                    } else {
                        test_X[n_][i] = (double) binomial(1, pNoise_Test, rng) * rng.nextDouble() * .5 + .5;
                    }
                }

                for (int i = 0; i < nOut; i++) {
                    if (i == pattern) {
                        test_T[n_][i] = 1;
                    } else {
                        test_T[n_][i] = 0;
                    }
                }
            }
        }


        // create minibatches
        for (int j = 0; j < minibatchSize; j++) {
            for (int i = 0; i < minibatch_N; i++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
                train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
            }
        }


        //
        // Build Dropout model
        //

        // construct Dropout
        System.out.print("Building the model...");
        Dropout classifier = new Dropout(nIn, hiddenLayerSizes, nOut, rng, "ReLU");
        System.out.println("done.");


        // train the model
        System.out.print("Training the model...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatch_N; batch++) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate, pDropout);
            }
            learningRate *= 0.999;
        }
        System.out.println("done.");


        // adjust the weight for testing
        System.out.print("Optimizing weights before testing...");
        classifier.pretest(pDropout);
        System.out.println("done.");


        // test
        for (int i = 0; i < test_N; i++) {
            predicted_T[i] = classifier.predict(test_X[i]);
        }


        //
        // Evaluate the model
        //

        int[][] confusionMatrix = new int[patterns][patterns];
        double accuracy = 0.;
        double[] precision = new double[patterns];
        double[] recall = new double[patterns];

        for (int i = 0; i < test_N; i++) {
            int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
            int actual_ = Arrays.asList(test_T[i]).indexOf(1);

            confusionMatrix[actual_][predicted_] += 1;
        }

        for (int i = 0; i < patterns; i++) {
            double col_ = 0.;
            double row_ = 0.;

            for (int j = 0; j < patterns; j++) {

                if (i == j) {
                    accuracy += confusionMatrix[i][j];
                    precision[i] += confusionMatrix[j][i];
                    recall[i] += confusionMatrix[i][j];
                }

                col_ += confusionMatrix[j][i];
                row_ += confusionMatrix[i][j];
            }
            precision[i] /= col_;
            recall[i] /= row_;
        }

        accuracy /= test_N;

        System.out.println("------------------------");
        System.out.println("Dropout model evaluation");
        System.out.println("------------------------");
        System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
        System.out.println("Precision:");
        for (int i = 0; i < patterns; i++) {
            System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
        }
        System.out.println("Recall:");
        for (int i = 0; i < patterns; i++) {
            System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
        }

    }
}
