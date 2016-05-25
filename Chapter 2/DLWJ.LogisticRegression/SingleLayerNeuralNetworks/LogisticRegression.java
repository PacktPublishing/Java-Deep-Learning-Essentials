package DLWJ.SingleLayerNeuralNetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import DLWJ.util.GaussianDistribution;
import static DLWJ.util.ActivationFunction.softmax;


public class LogisticRegression {

    public int nIn;
    public int nOut;
    public double[][] W;
    public double[] b;


    public LogisticRegression(int nIn, int nOut) {

        this.nIn = nIn;
        this.nOut = nOut;

        W = new double[nOut][nIn];
        b = new double[nOut];

    }

    public double[][] train(double[][] X, int T[][], int minibatchSize, double learningRate) {

        double[][] grad_W = new double[nOut][nIn];
        double[] grad_b = new double[nOut];

        double[][] dY = new double[minibatchSize][nOut];

        // train with SGD
        // 1. calculate gradient of W, b
        for (int n = 0; n < minibatchSize; n++) {

            double[] predicted_Y_ = output(X[n]);

            for (int j = 0; j < nOut; j++) {
                dY[n][j] = predicted_Y_[j] - T[n][j];

                for (int i = 0; i < nIn; i++) {
                    grad_W[j][i] += dY[n][j] * X[n][i];
                }

                grad_b[j] += dY[n][j];
            }
        }

        // 2. update params
        for (int j = 0; j < nOut; j++) {
            for (int i = 0; i < nIn; i++) {
                W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
            }
            b[j] -= learningRate * grad_b[j] / minibatchSize;
        }

        return dY;
    }

    public double[] output(double[] x) {

        double[] preActivation = new double[nOut];

        for (int j = 0; j < nOut; j++) {

            for (int i = 0; i < nIn; i++) {
                preActivation[j] += W[j][i] * x[i];
            }

            preActivation[j] += b[j];  // linear output
        }

        return softmax(preActivation, nOut);
    }

    public Integer[] predict(double[] x) {

        double[] y = output(x);  // activate input data through learned networks
        Integer[] t = new Integer[nOut]; // output is the probability, so cast it to label

        int argmax = -1;
        double max = 0.;

        for (int i = 0; i < nOut; i++) {
            if (max < y[i]) {
                max = y[i];
                argmax = i;
            }
        }

        for (int i = 0; i < nOut; i++) {
            if (i == argmax) {
                t[i] = 1;
            } else {
                t[i] = 0;
            }
        }

        return t;
    }


    public static void main(String[] args) {

        final Random rng = new Random(1234);  // seed random

        //
        // Declare variables and constants
        //

        final int patterns = 3;  // number of classes
        final int train_N = 400 * patterns;
        final int test_N = 60 * patterns;
        final int nIn = 2;
        final int nOut = patterns;

        double[][] train_X = new double[train_N][nIn];
        int[][] train_T = new int[train_N][nOut];

        double[][] test_X = new double[test_N][nIn];
        Integer[][] test_T = new Integer[test_N][nOut];
        Integer[][] predicted_T = new Integer[test_N][nOut];

        int epochs = 2000;
        double learningRate = 0.2;

        int minibatchSize = 50;  //  number of data in each minibatch
        int minibatch_N = train_N / minibatchSize; //  number of minibatches

        double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];  // minibatches of training data
        int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];       // minibatches of output data for training
        List<Integer> minibatchIndex = new ArrayList<>();  // data index for minibatch to apply SGD
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);  // shuffle data index for SGD


        //
        // Training data for demo
        //   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
        //   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
        //   class 3 : x3 ~ N(  0.0, 1.0 ), y3 ~ N(  0.0, 1.0 )
        //

        GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
        GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);
        GaussianDistribution g3 = new GaussianDistribution(0.0, 1.0, rng);

        // data set in class 1
        for (int i = 0; i < train_N/patterns - 1; i++) {
            train_X[i][0] = g1.random();
            train_X[i][1] = g2.random();
            train_T[i] = new int[]{1, 0, 0};
        }
        for (int i = 0; i < test_N/patterns - 1; i++) {
            test_X[i][0] = g1.random();
            test_X[i][1] = g2.random();
            test_T[i] = new Integer[]{1, 0, 0};
        }

        // data set in class 2
        for (int i = train_N/patterns - 1; i < train_N/patterns * 2 - 1; i++) {
            train_X[i][0] = g2.random();
            train_X[i][1] = g1.random();
            train_T[i] = new int[]{0, 1, 0};
        }
        for (int i = test_N/patterns - 1; i < test_N/patterns * 2 - 1; i++) {
            test_X[i][0] = g2.random();
            test_X[i][1] = g1.random();
            test_T[i] = new Integer[]{0, 1, 0};
        }

        // data set in class 3
        for (int i = train_N/patterns * 2 - 1; i < train_N; i++) {
            train_X[i][0] = g3.random();
            train_X[i][1] = g3.random();
            train_T[i] = new int[]{0, 0, 1};
        }
        for (int i = test_N/patterns * 2 - 1; i < test_N; i++) {
            test_X[i][0] = g3.random();
            test_X[i][1] = g3.random();
            test_T[i] = new Integer[]{0, 0, 1};
        }

        // create minibatches with training data
        for (int i = 0; i < minibatch_N; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
                train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
            }
        }


        //
        // Build Logistic Regression model
        //

        // construct logistic regression
        LogisticRegression classifier = new LogisticRegression(nIn, nOut);

        // train
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatch_N; batch++) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
            }
            learningRate *= 0.95;
        }


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

        System.out.println("------------------------------------");
        System.out.println("Logistic Regression model evaluation");
        System.out.println("------------------------------------");
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
