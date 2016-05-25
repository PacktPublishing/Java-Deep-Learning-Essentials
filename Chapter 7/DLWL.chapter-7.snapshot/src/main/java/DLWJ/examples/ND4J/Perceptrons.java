package DLWJ.examples.ND4J;

import java.util.Random;
import DLWJ.util.GaussianDistribution;
import static DLWJ.util.ActivationFunction.step;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class Perceptrons {

    public int nIn;       // dimensions of input data
    public INDArray w;


    public Perceptrons(int nIn) {

        this.nIn = nIn;
        w = Nd4j.create(new double[nIn], new int[]{nIn, 1});

    }

    public int train(INDArray x, INDArray t, double learningRate) {

        int classified = 0;

        // check if the data is classified correctly
        double c = x.mmul(w).getDouble(0) * t.getDouble(0);

        // apply steepest descent method if the data is wrongly classified
        if (c > 0) {
            classified = 1;
        } else {
            w.addi(x.transpose().mul(t).mul(learningRate));
        }

        return classified;
    }

    public int predict(INDArray x) {

        return step(x.mmul(w).getDouble(0));
    }


    public static void main(String[] args) {

        //
        // Declare (Prepare) variables and constants for perceptrons
        //

        final int train_N = 1000;  // number of training data
        final int test_N = 200;   // number of test data
        final int nIn = 2;        // dimensions of input data

        INDArray train_X = Nd4j.create(new double[train_N * nIn], new int[]{train_N, nIn});  // input data for training
        INDArray train_T = Nd4j.create(new double[train_N], new int[]{train_N, 1});          // output data (label) for training

        INDArray test_X = Nd4j.create(new double[test_N * nIn], new int[]{test_N, nIn});  // input data for test
        INDArray test_T = Nd4j.create(new double[test_N], new int[]{test_N, 1});          // label of inputs
        INDArray predicted_T = Nd4j.create(new double[test_N], new int[]{test_N, 1});     // output data predicted by the model


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
            train_X.put(i, 0, Nd4j.scalar(g1.random()));
            train_X.put(i, 1, Nd4j.scalar(g2.random()));
            train_T.put(i, Nd4j.scalar(1));
        }
        for (int i = 0; i < test_N/2 - 1; i++) {
            test_X.put(i, 0, Nd4j.scalar(g1.random()));
            test_X.put(i, 1, Nd4j.scalar(g2.random()));
            test_T.put(i, Nd4j.scalar(1));
        }

        // data set in class 2
        for (int i = train_N/2; i < train_N; i++) {
            train_X.put(i, 0, Nd4j.scalar(g2.random()));
            train_X.put(i, 1, Nd4j.scalar(g1.random()));
            train_T.put(i, Nd4j.scalar(-1));
        }
        for (int i = test_N/2; i < test_N; i++) {
            test_X.put(i, 0, Nd4j.scalar(g2.random()));
            test_X.put(i, 1, Nd4j.scalar(g1.random()));
            test_T.put(i, Nd4j.scalar(-1));
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
                classified_ += classifier.train(train_X.getRow(i), train_T.getRow(i), learningRate);
            }

            if (classified_ == train_N) break;  // when all data classified correctly

            epoch++;
            if (epoch > epochs) break;
        }


        // test
        for (int i = 0; i < test_N; i++) {
            predicted_T.put(i, Nd4j.scalar(classifier.predict(test_X.getRow(i))));
        }


        //
        // Evaluate the model
        //

        int[][] confusionMatrix = new int[2][2];
        double accuracy = 0.;
        double precision = 0.;
        double recall = 0.;

        for (int i = 0; i < test_N; i++) {

            if (predicted_T.getRow(i).getDouble(0) > 0) {
                if (test_T.getRow(i).getDouble(0) > 0) {
                    accuracy += 1;
                    precision += 1;
                    recall += 1;
                    confusionMatrix[0][0] += 1;
                } else {
                    confusionMatrix[1][0] += 1;
                }
            } else {
                if (test_T.getRow(i).getDouble(0) > 0) {
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

    }}
