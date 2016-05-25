package DLWJ.util;

import java.util.Random;


public final class GaussianDistribution {

    private final double mean;
    private final double var;
    private final Random rng;

    public GaussianDistribution(double mean, double var, Random rng) {
        if (var < 0.0) {
            throw new IllegalArgumentException("Variance must be non-negative value.");
        }

        this.mean = mean;
        this.var = var;

        if (rng == null) {
            rng = new Random();
        }
        this.rng = rng;
    }

    public double random() {
        double r = 0.0;
        while (r == 0.0) {
            r = rng.nextDouble();
        }

        double c = Math.sqrt( -2.0 * Math.log(r) );

        if (rng.nextDouble() < 0.5) {
            return c * Math.sin( 2.0 * Math.PI * rng.nextDouble() ) * var + mean;
        }

        return c * Math.cos( 2.0 * Math.PI * rng.nextDouble() ) * var + mean;
    }

}
