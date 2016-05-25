package DLWJ.DeepNeuralNetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import DLWJ.MultiLayerNeuralNetworks.HiddenLayer;
import DLWJ.SingleLayerNeuralNetworks.LogisticRegression;
import static DLWJ.util.RandomGenerator.binomial;


public class StackedDenoisingAutoencoders {

    public int nIn;
    public int[] hiddenLayerSizes;
    public int nOut;
    public int nLayers;
    public DenoisingAutoencoders[] daLayers;
    public HiddenLayer[] sigmoidLayers;
    public LogisticRegression logisticLayer;
    public Random rng;


    public StackedDenoisingAutoencoders(int nIn, int[] hiddenLayerSizes, int nOut, Random rng) {

        if (rng == null) rng = new Random(1234);

        this.nIn = nIn;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.nOut = nOut;
        this.nLayers = hiddenLayerSizes.length;
        this.sigmoidLayers = new HiddenLayer[nLayers];
        this.daLayers = new DenoisingAutoencoders[nLayers];
        this.rng = rng;

        // construct multi-layer
        for (int i = 0; i < nLayers; i++) {
            int nIn_;
            if (i == 0) nIn_ = nIn;
            else nIn_ = hiddenLayerSizes[i-1];

            // construct hidden layers with sigmoid function
            //   weight matrices and bias vectors will be shared with RBM layers
            sigmoidLayers[i] = new HiddenLayer(nIn_, hiddenLayerSizes[i], null, null, rng, "sigmoid");

            // construct DA layers
            daLayers[i] = new DenoisingAutoencoders(nIn_, hiddenLayerSizes[i], sigmoidLayers[i].W, sigmoidLayers[i].b, null, rng);
        }

        // logistic regression layer for output
        logisticLayer = new LogisticRegression(hiddenLayerSizes[nLayers-1], nOut);
    }

    public void pretrain(double[][][] X, int minibatchSize, int minibatch_N, int epochs, double learningRate, double corruptionLevel) {

        for (int layer = 0; layer < nLayers; layer++) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int batch = 0; batch < minibatch_N; batch++) {

                    double[][] X_ = new double[minibatchSize][nIn];
                    double[][] prevLayerX_;

                    // Set input data for current layer
                    if (layer == 0) {
                        X_ = X[batch];
                    } else {

                        prevLayerX_ = X_;
                        X_ = new double[minibatchSize][hiddenLayerSizes[layer-1]];

                        for (int i = 0; i < minibatchSize; i++) {
                            X_[i] = sigmoidLayers[layer-1].output(prevLayerX_[i]);
                        }
                    }

                    daLayers[layer].train(X_, minibatchSize, learningRate, corruptionLevel);
                }
            }
        }

    }

    public void finetune(double[][] X, int[][] T, int minibatchSize, double learningRate) {

        List<double[][]> layerInputs = new ArrayList<>(nLayers + 1);
        layerInputs.add(X);

        double[][] Z = new double[0][0];
        double[][] dY;

        // forward hidden layers
        for (int layer = 0; layer < nLayers; layer++) {

            double[] x_;  // layer input
            double[][] Z_ = new double[minibatchSize][hiddenLayerSizes[layer]];

            for (int n = 0; n < minibatchSize; n++) {

                if (layer == 0) {
                    x_ = X[n];
                } else {
                    x_ = Z[n];
                }

                Z_[n] = sigmoidLayers[layer].forward(x_);
            }

            Z = Z_;
            layerInputs.add(Z.clone());
        }

        // forward & backward output layer
        dY = logisticLayer.train(Z, T, minibatchSize, learningRate);

        // backward hidden layers
        double[][] Wprev;
        double[][] dZ = new double[0][0];

        for (int layer = nLayers - 1; layer >= 0; layer--) {

            if (layer == nLayers - 1) {
                Wprev = logisticLayer.W;
            } else {
                Wprev = sigmoidLayers[layer+1].W;
                dY = dZ.clone();
            }

            dZ = sigmoidLayers[layer].backward(layerInputs.get(layer), layerInputs.get(layer+1), dY, Wprev, minibatchSize, learningRate);
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

            z = sigmoidLayers[layer].forward(x_);
        }

        return logisticLayer.predict(z);
    }



    public static void main(String[] args) {

        final Random rng = new Random(123);

        //
        // Declare variables and constants
        //

        int train_N_each = 200;        // for demo
        int validation_N_each = 200;   // for demo
        int test_N_each = 50;          // for demo
        int nIn_each = 20;             // for demo
        double pNoise_Training = 0.2;  // for demo
        double pNoise_Test = 0.25;     // for demo

        final int patterns = 3;

        final int train_N = train_N_each * patterns;
        final int validation_N = validation_N_each * patterns;
        final int test_N = test_N_each * patterns;

        final int nIn = nIn_each * patterns;
        final int nOut = patterns;
        int[] hiddenLayerSizes = {20, 20};
        double corruptionLevel = 0.3;

        double[][] train_X = new double[train_N][nIn];

        double[][] validation_X = new double[validation_N][nIn];
        int[][] validation_T = new int[validation_N][nOut];

        double[][] test_X = new double[test_N][nIn];
        Integer[][] test_T = new Integer[test_N][nOut];
        Integer[][] predicted_T = new Integer[test_N][nOut];

        int pretrainEpochs = 1000;
        double pretrainLearningRate = 0.2;
        int finetuneEpochs = 1000;
        double finetuneLearningRate = 0.15;

        int minibatchSize = 50;
        final int train_minibatch_N = train_N / minibatchSize;
        final int validation_minibatch_N = validation_N / minibatchSize;

        double[][][] train_X_minibatch = new double[train_minibatch_N][minibatchSize][nIn];
        double[][][] validation_X_minibatch = new double[validation_minibatch_N][minibatchSize][nIn];
        int[][][] validation_T_minibatch = new int[validation_minibatch_N][minibatchSize][nOut];
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
            }

            for (int n = 0; n < validation_N_each; n++) {

                int n_ = pattern * validation_N_each + n;

                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= validation_N_each * pattern && n_ < validation_N_each * (pattern + 1) ) &&
                            (i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
                        validation_X[n_][i] = (double) binomial(1, 1 - pNoise_Training, rng) * rng.nextDouble() * .5 + .5;
                    } else {
                        validation_X[n_][i] = (double) binomial(1, pNoise_Training, rng) * rng.nextDouble() * .5 + .5;
                    }
                }

                for (int i = 0; i < nOut; i++) {
                    if (i == pattern) {
                        validation_T[n_][i] = 1;
                    } else {
                        validation_T[n_][i] = 0;
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
            for (int i = 0; i < train_minibatch_N; i++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
            }
            for (int i = 0; i < validation_minibatch_N; i++) {
                validation_X_minibatch[i][j] = validation_X[minibatchIndex.get(i * minibatchSize + j)];
                validation_T_minibatch[i][j] = validation_T[minibatchIndex.get(i * minibatchSize + j)];
            }
        }


        //
        // Build Stacked Denoising Autoencoders model
        //

        // construct SDA
        System.out.print("Building the model...");
        StackedDenoisingAutoencoders classifier = new StackedDenoisingAutoencoders(nIn, hiddenLayerSizes, nOut, rng);
        System.out.println("done.");


        // pre-training the model
        System.out.print("Pre-training the model...");
        classifier.pretrain(train_X_minibatch, minibatchSize, train_minibatch_N, pretrainEpochs, pretrainLearningRate, corruptionLevel);
        System.out.println("done.");


        // fine-tuning the model
        System.out.print("Fine-tuning the model...");
        for (int epoch = 0; epoch < finetuneEpochs; epoch++) {
            for (int batch = 0; batch < validation_minibatch_N; batch++) {
                classifier.finetune(validation_X_minibatch[batch], validation_T_minibatch[batch], minibatchSize, finetuneLearningRate);
            }
            finetuneLearningRate *= 0.98;
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
        System.out.println("SDA model evaluation");
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
