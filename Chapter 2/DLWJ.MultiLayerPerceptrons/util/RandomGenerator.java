package DLWJ.util;

import java.util.Random;


public final class RandomGenerator {

    public static double uniform(double min, double max, Random rng) {
        return rng.nextDouble() * (max - min) + min;
    }

}
