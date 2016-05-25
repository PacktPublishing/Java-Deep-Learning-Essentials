package DLWJ.DeepNeuralNetworks;

import java.util.*;

import DLWJ.MultiLayerNeuralNetworks.HiddenLayer;
import DLWJ.SingleLayerNeuralNetworks.LogisticRegression;
import static DLWJ.util.RandomGenerator.binomial;


public class ConvolutionalNeuralNetworks {

    public int[] nKernels;
    public int[][] kernelSizes;
    public int[][] poolSizes;
    public int nHidden;
    public int nOut;

    public ConvolutionPoolingLayer[] convpoolLayers;
    public int[][] convolvedSizes;
    public int[][] pooledSizes;
    public int flattenedSize;
    public HiddenLayer hiddenLayer;
    public LogisticRegression logisticLayer;
    public Random rng;


    public ConvolutionalNeuralNetworks(int[] imageSize, int channel, int[] nKernels, int[][] kernelSizes, int[][] poolSizes, int nHidden, int nOut, Random rng, String activation) {

        if (rng == null) rng = new Random(1234);

        this.nKernels = nKernels;
        this.kernelSizes = kernelSizes;
        this.poolSizes = poolSizes;
        this.nHidden = nHidden;
        this.nOut = nOut;
        this.rng = rng;

        convpoolLayers = new ConvolutionPoolingLayer[nKernels.length];
        convolvedSizes = new int[nKernels.length][imageSize.length];
        pooledSizes = new int[nKernels.length][imageSize.length];


        // construct convolution + pooling layers
        for (int i = 0; i < nKernels.length; i++) {
            int[] size_;
            int channel_;

            if (i == 0) {
                size_ = new int[]{imageSize[0], imageSize[1]};
                channel_ = channel;
            } else {
                size_ = new int[]{pooledSizes[i-1][0], pooledSizes[i-1][1]};
                channel_ = nKernels[i-1];
            }

            convolvedSizes[i] = new int[]{size_[0] - kernelSizes[i][0] + 1, size_[1] - kernelSizes[i][1] + 1};
            pooledSizes[i] = new int[]{convolvedSizes[i][0] / poolSizes[i][0], convolvedSizes[i][1] / poolSizes[i][0]};

            convpoolLayers[i] = new ConvolutionPoolingLayer(size_, channel_, nKernels[i], kernelSizes[i], poolSizes[i], convolvedSizes[i], pooledSizes[i], rng, activation);
        }


        // build MLP
        flattenedSize = nKernels[nKernels.length-1] * pooledSizes[pooledSizes.length-1][0] * pooledSizes[pooledSizes.length-1][1];

        // construct hidden layer
        hiddenLayer = new HiddenLayer(flattenedSize, nHidden, null, null, rng, activation);

        // construct output layer
        logisticLayer = new LogisticRegression(nHidden, nOut);
    }


    public void train(double[][][][] X, int[][] T, int minibatchSize, double learningRate) {

        // cache pre-activated, activated, and downsampled inputs of each convolution + pooling layer for backpropagation
        List<double[][][][]> preActivated_X = new ArrayList<>(nKernels.length);
        List<double[][][][]> activated_X = new ArrayList<>(nKernels.length);
        List<double[][][][]> downsampled_X = new ArrayList<>(nKernels.length+1);  // +1 for input X
        downsampled_X.add(X);

        for (int i = 0; i < nKernels.length; i++) {
            preActivated_X.add(new double[minibatchSize][nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
            activated_X.add(new double[minibatchSize][nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
            downsampled_X.add(new double[minibatchSize][nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
        }

        double[][] flattened_X = new double[minibatchSize][flattenedSize];  // cache flattened inputs

        double[][] Z = new double[minibatchSize][nHidden];  // cache outputs of hidden layer

        double[][] dY;  // delta of output layer
        double[][] dZ;  // delta of hidden layer
        double[][] dX_flatten = new double[minibatchSize][flattenedSize];  // delta of input layer
        double[][][][] dX = new double[minibatchSize][nKernels[nKernels.length-1]][pooledSizes[pooledSizes.length-1][0]][pooledSizes[pooledSizes.length-1][1]];

        double[][][][] dC;


        // train with minibatch
        for (int n = 0; n < minibatchSize; n++) {

            // forward convolution + pooling layers
            double[][][] z_ = X[n].clone();
            for (int i = 0; i < nKernels.length; i++) {
                z_ = convpoolLayers[i].forward(z_, preActivated_X.get(i)[n], activated_X.get(i)[n]);
                downsampled_X.get(i+1)[n] = z_.clone();
            }

            // flatten output to make it input for fully connected MLP
            double[] x_ = this.flatten(z_);
            flattened_X[n] = x_.clone();

            // forward hidden layer
            Z[n] = hiddenLayer.forward(x_);

        }


        // forward & backward output layer
        dY = logisticLayer.train(Z, T, minibatchSize, learningRate);

        // backward hidden layer
        dZ = hiddenLayer.backward(flattened_X, Z, dY, logisticLayer.W, minibatchSize, learningRate);

        // backpropagate delta to input layer
        for (int n = 0; n < minibatchSize; n++) {
            for (int i = 0; i < flattenedSize; i++) {
                for (int j = 0; j < nHidden; j++) {
                    dX_flatten[n][i] += hiddenLayer.W[j][i] * dZ[n][j];
                }
            }

            dX[n] = unflatten(dX_flatten[n]);  // unflatten delta
        }

        // backward convolution + pooling layers
        dC = dX.clone();
        for (int i = nKernels.length-1; i >= 0; i--) {
            dC = convpoolLayers[i].backward(downsampled_X.get(i), preActivated_X.get(i), activated_X.get(i), downsampled_X.get(i+1), dC, minibatchSize, learningRate);
        }

    }


    public double[] flatten(double[][][] z) {

        double[] x = new double[flattenedSize];

        int index = 0;
        for (int k = 0; k < nKernels[nKernels.length-1]; k++) {
            for (int i = 0; i < pooledSizes[pooledSizes.length-1][0]; i++) {
                for (int j = 0; j < pooledSizes[pooledSizes.length-1][1]; j++) {
                    x[index] = z[k][i][j];
                    index += 1;
                }
            }
        }

        return x;
    }

    public double[][][] unflatten(double[] x) {

        double[][][] z = new double[nKernels[nKernels.length-1]][pooledSizes[pooledSizes.length-1][0]][pooledSizes[pooledSizes.length-1][1]];

        int index = 0;
        for (int k = 0; k < z.length; k++) {
            for (int i = 0; i < z[0].length; i++) {
                for (int j = 0; j < z[0][0].length; j++) {
                    z[k][i][j] = x[index];
                    index += 1;
                }
            }
        }

        return z;
    }


    public Integer[] predict(double[][][] x) {

        List<double[][][]> preActivated = new ArrayList<>(nKernels.length);
        List<double[][][]> activated = new ArrayList<>(nKernels.length);

        for (int i = 0; i < nKernels.length; i++) {
            preActivated.add(new double[nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
            activated.add(new double[nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
        }

        // forward convolution + pooling layers
        double[][][] z = x.clone();
        for (int i = 0; i < nKernels.length; i++) {
            z = convpoolLayers[i].forward(z, preActivated.get(i), activated.get(i));
        }


        // forward MLP
        return logisticLayer.predict(hiddenLayer.forward(this.flatten(z)));
    }

    public static void main(String[] args) {

        final Random rng = new Random(123);  // seed random

        //
        // Declare variables and constants
        //

        int train_N_each = 50;        // for demo
        int test_N_each = 10;          // for demo
        double pNoise_Training = 0.05;  // for demo
        double pNoise_Test = 0.10;     // for demo

        final int patterns = 3;

        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;

        final int[] imageSize = {12, 12};
        final int channel = 1;

        int[] nKernels = {10, 20};
        int[][] kernelSizes = { {3, 3}, {2, 2} };
        int[][] poolSizes = { {2, 2}, {2, 2} };

        int nHidden = 20;
        final int nOut = patterns;

        double[][][][] train_X = new double[train_N][channel][imageSize[0]][imageSize[1]];
        int[][] train_T = new int[train_N][nOut];

        double[][][][] test_X = new double[test_N][channel][imageSize[0]][imageSize[1]];
        Integer[][] test_T = new Integer[test_N][nOut];
        Integer[][] predicted_T = new Integer[test_N][nOut];


        int epochs = 500;
        double learningRate = 0.1;

        final int minibatchSize = 25;
        int minibatch_N = train_N / minibatchSize;

        double[][][][][] train_X_minibatch = new double[minibatch_N][minibatchSize][channel][imageSize[0]][imageSize[1]];
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

                for (int c = 0; c < channel; c++) {

                    for (int i = 0; i < imageSize[0]; i++) {

                        for (int j = 0; j < imageSize[1]; j++) {

                            if ((i < (pattern + 1) * (imageSize[0] / patterns)) && (i >= pattern * imageSize[0] / patterns)) {
                                train_X[n_][c][i][j] = ((int) 128. * rng.nextDouble() + 128.) * binomial(1, 1 - pNoise_Training, rng) / 256.;
                            } else {
                                train_X[n_][c][i][j] = 128. * binomial(1, pNoise_Training, rng) / 256.;
                            }
                        }
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

                for (int c = 0; c < channel; c++) {

                    for (int i = 0; i < imageSize[0]; i++) {

                        for (int j = 0; j < imageSize[1]; j++) {

                            if ((i < (pattern + 1) * imageSize[0] / patterns) && (i >= pattern * imageSize[0] / patterns)) {
                                test_X[n_][c][i][j] = ((int) 128. * rng.nextDouble() + 128.) * binomial(1, 1 - pNoise_Test, rng) / 256.;
                            } else {
                                test_X[n_][c][i][j] = 128. * binomial(1, pNoise_Test, rng) / 256.;
                            }
                        }
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
        // Build Convolutional Neural Networks model
        //

        // construct CNN
        System.out.print("Building the model...");
        ConvolutionalNeuralNetworks classifier = new ConvolutionalNeuralNetworks(imageSize, channel, nKernels, kernelSizes, poolSizes, nHidden, nOut, rng, "ReLU");
        System.out.println("done.");


        // train the model
        System.out.print("Training the model...");
        System.out.println();

        for (int epoch = 0; epoch < epochs; epoch++) {

            if ((epoch + 1) % 50 == 0) {
                System.out.println("\titer = " + (epoch + 1) + " / " + epochs);
            }

            for (int batch = 0; batch < minibatch_N; batch++) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
            }
            learningRate *= 0.999;
        }
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

        System.out.println("--------------------");
        System.out.println("CNN model evaluation");
        System.out.println("--------------------");
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
