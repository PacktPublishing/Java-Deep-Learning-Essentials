package DLWJ.MultiLayerNeuralNetworks;

import java.util.Random;
import java.util.function.DoubleFunction;
import static DLWJ.util.ActivationFunction.*;
import static DLWJ.util.RandomGenerator.uniform;


public class HiddenLayer {

    public int nIn;
    public int nOut;
    public double[][] W;
    public double[] b;
    public Random rng;
    public DoubleFunction<Double> activation;
    public DoubleFunction<Double> dactivation;

    public HiddenLayer(int nIn, int nOut, double[][] W, double[] b, Random rng, String activation) {

        if (rng == null) rng = new Random(1234);  // seed random

        if (W == null) {

            W = new double[nOut][nIn];
            double w_ = 1. / nIn;

            for(int j = 0; j < nOut; j++) {
                for(int i = 0; i < nIn; i++) {
                    W[j][i] = uniform(-w_, w_, rng);  // initialize W with uniform distribution
                }
            }

        }

        if (b == null) b = new double[nOut];

        this.nIn = nIn;
        this.nOut = nOut;
        this.W = W;
        this.b = b;
        this.rng = rng;

        if (activation == "sigmoid" || activation == null) {

            this.activation = (double x) -> sigmoid(x);
            this.dactivation = (double x) -> dsigmoid(x);

        } else if (activation == "tanh") {

            this.activation = (double x) -> tanh(x);
            this.dactivation = (double x) -> dtanh(x);

        }  else {
            throw new IllegalArgumentException("activation function not supported");
        }

    }

    public double[] output(double[] x) {

        double[] y = new double[nOut];

        for (int j = 0; j < nOut; j++) {
            double preActivation_ = 0.;

            for (int i = 0; i < nIn; i++) {
                preActivation_ += W[j][i] * x[i];
            }
            preActivation_ += b[j];

            y[j] = activation.apply(preActivation_);
        }

        return y;
    }

    public double[] forward(double[] x) {
        return output(x);
    }

    public double[][] backward(double[][] X, double[][] Z, double[][] dY, double[][] Wprev, int minibatchSize, double learningRate) {

        double[][] dZ = new double[minibatchSize][nOut];  // backpropagation error

        double[][] grad_W = new double[nOut][nIn];
        double[] grad_b = new double[nOut];

        // train with SGD
        // calculate backpropagation error to get gradient of W, b
        for (int n = 0; n < minibatchSize; n++) {

            for (int j = 0; j < nOut; j++) {

                for (int k = 0; k < dY[0].length; k++) {  // k < ( nOut of previous layer )
                    dZ[n][j] += Wprev[k][j] * dY[n][k];
                }
                dZ[n][j] *= dactivation.apply(Z[n][j]);


                for (int i = 0; i < nIn; i++) {
                    grad_W[j][i] += dZ[n][j] * X[n][i];
                }

                grad_b[j] += dZ[n][j];
            }
        }

        // update params
        for (int j = 0; j < nOut; j++) {
            for(int i = 0; i < nIn; i++) {
                W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
            }
            b[j] -= learningRate * grad_b[j] / minibatchSize;
        }

        return dZ;
    }

}
