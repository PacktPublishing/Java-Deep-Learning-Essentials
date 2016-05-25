package DLWJ.util;


public final class ActivationFunction {

    public static int step(double x) {
        if (x >= 0) {
            return 1;
        } else {
            return -1;
        }
    }

}
