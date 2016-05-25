package DNN.SingleLayerNeuralNetworks;

import java.util.Random;
import DNN.util.GaussianDistribution;
import static DNN.util.ActivationFunction.step;


public class Perceptrons {

    public int nIn;       // dimensions of input data
    public double[] w;  // weight vector of perceptrons


    public Perceptrons(int nIn) {

        this.nIn = nIn;
        w = new double[nIn];

    }

    public int train(double[] x, int t, double learningRate) {

        int classified = 0;
        double c = 0.;

        // check if the data is classified correctly
        for (int i = 0; i < nIn; i++) {
            c += w[i] * x[i] * t;
        }

        // apply steepest descent method if the data is wrongly classified
        if (c > 0) {
            classified = 1;
        } else {
            for (int i = 0; i < nIn; i++) {
                w[i] += learningRate * x[i] * t;
            }
        }

        return classified;
    }

    public int predict (double[] x) {

        double preActivation = 0.;

        for (int i = 0; i < nIn; i++) {
            preActivation += w[i] * x[i];
        }

        return step(preActivation);
    }


    public static void main(String[] args) {

        //
        // Declare (Prepare) variables and constants for perceptrons
        //

        final int train_N = 1000;  // number of training data
        final int test_N = 200;   // number of test data
        final int nIn = 2;        // dimensions of input data

        double[][] train_X = new double[train_N][nIn];  // input data for training
        int[] train_T = new int[train_N];               // output data (label) for training

        double[][] test_X = new double[test_N][nIn];  // input data for test
        int[] test_T = new int[test_N];               // label of inputs
        int[] predicted_T = new int[test_N];          // output data predicted by the model

        final int epochs = 2000;   // maximum training epochs
        final double learningRate = 1.;  // learning rate can be 1 in perceptrons


        //
        // Create training data and test data for demo.
        //
        // Let training data set for each class follow Normal (Gaussian) distribution here:
        //   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
        //   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
        //

        final Random rng = new Random(1234);  // seed random
        GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
        GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);

        // data set in class 1
        for (int i = 0; i < train_N/2 - 1; i++) {
            train_X[i][0] = g1.random();
            train_X[i][1] = g2.random();
            train_T[i] = 1;
        }
        for (int i = 0; i < test_N/2 - 1; i++) {
            test_X[i][0] = g1.random();
            test_X[i][1] = g2.random();
            test_T[i] = 1;
        }

        // data set in class 2
        for (int i = train_N/2; i < train_N; i++) {
            train_X[i][0] = g2.random();
            train_X[i][1] = g1.random();
            train_T[i] = -1;
        }
        for (int i = test_N/2; i < test_N; i++) {
            test_X[i][0] = g2.random();
            test_X[i][1] = g1.random();
            test_T[i] = -1;
        }


        //
        // Build SingleLayerNeuralNetworks model
        //

        int epoch = 0;  // training epochs

        // construct perceptrons
        Perceptrons classifier = new Perceptrons(nIn);

        // train models
        while (true) {
            int classified_ = 0;

            for (int i=0; i < train_N; i++) {
                classified_ += classifier.train(train_X[i], train_T[i], learningRate);
            }

            if (classified_ == train_N) break;  // when all data classified correctly

            epoch++;
            if (epoch > epochs) break;
        }


        // test
        for (int i = 0; i < test_N; i++) {
            predicted_T[i] = classifier.predict(test_X[i]);
        }


        //
        // Evaluate the model
        //

        int[][] confusionMatrix = new int[2][2];
        double accuracy = 0.;
        double precision = 0.;
        double recall = 0.;

        for (int i = 0; i < test_N; i++) {

            if (predicted_T[i] > 0) {
                if (test_T[i] > 0) {
                    accuracy += 1;
                    precision += 1;
                    recall += 1;
                    confusionMatrix[0][0] += 1;
                } else {
                    confusionMatrix[1][0] += 1;
                }
            } else {
                if (test_T[i] > 0) {
                    confusionMatrix[0][1] += 1;
                } else {
                    accuracy += 1;
                    confusionMatrix[1][1] += 1;
                }
            }

        }

        accuracy /= test_N;
        precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
        recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

        System.out.println("----------------------------");
        System.out.println("Perceptrons model evaluation");
        System.out.println("----------------------------");
        System.out.printf("Accuracy:  %.1f %%\n", accuracy * 100);
        System.out.printf("Precision: %.1f %%\n", precision * 100);
        System.out.printf("Recall:    %.1f %%\n", recall * 100);

    }
}
