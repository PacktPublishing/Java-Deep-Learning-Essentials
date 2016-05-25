package DLWJ.SingleLayerNeuralNetworks;

import java.util.*;

public class LogisticRegressionXOR {
    public static void main(String[] args) {

        final Random rng = new Random(1234);  // seed random

        //
        // Declare variables and constants
        //

        final int patterns = 2;  // number of classes
        final int train_N = 4;
        final int test_N = 4;
        final int nIn = 2;
        final int nOut = patterns;

        double[][] train_X;
        int[][] train_T;

        double[][] test_X;
        Integer[][] test_T;
        Integer[][] predicted_T = new Integer[test_N][nOut];

        final int epochs = 2000;
        double learningRate = 0.2;

        int minibatchSize = 1;  //  set 1 for on-line training
        int minibatch_N = train_N / minibatchSize;

        double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];
        int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);

        //
        // Training data for demo
        //   class 1 : [0, 0], [1, 1]  for negative class
        //   class 2 : [0, 1], [1, 0]  for positive class
        //

        train_X = new double[][]{
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
        };
        train_T = new int[][]{
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1}
        };
        test_X = new double[][]{
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
        };
        test_T = new Integer[][]{
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1}
        };

        // create minibatches
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

        // output
        for (int i = 0; i < test_N; i++) {
            System.out.print("[" + test_X[i][0] + ", " + test_X[i][1] + "] -> Prediction: ");

            if (predicted_T[i][0] > predicted_T[i][1]) {
                System.out.print("Positive, ");
                System.out.print("probability = " + predicted_T[i][0]);
            } else {
                System.out.print("Negative, ");
                System.out.print("probability = " + predicted_T[i][1]);
            }

            System.out.print("; Actual: ");
            if (test_T[i][0] == 1) {
                System.out.println("Positive");
            } else {
                System.out.println("Negative");
            }
        }

    }
}
