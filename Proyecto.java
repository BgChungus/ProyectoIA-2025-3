
import java.util.Arrays;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import java.io.File;  // Import the File class
import java.io.IOException;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.util.Random;


class Image {
    int width, height;
    double[] data;

    Image(int width, int height, double[] data) {
        this.width = width;
        this.height = height;
        this.data = data;
    }
}

public class Proyecto {
    private static final int EPOCHS = 100;
    private static final double LEARNING_RATE = 0.1;
    private static final Random RANDOM = new Random(42); // Seed for reproducibility

    // Utilidades Matematicas

    public static double relu(double x) {
        return Math.max(0, x);
    }
    public static double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    public static double sigmoidDerivative(double output) {
        return output * (1.0 - output);
    }

    public static double mseLoss(double predicted, double target) {
        return 0.5 * Math.pow(predicted - target, 2);
    }
    public static double mseLossDerivative(double predicted, double target) {
        return predicted - target;
    }

    static class ConvolutionalLayer {
        private final double[][] filter; // 3x3 Filter (Weight)
        private double bias = RANDOM.nextDouble() * 0.1;
        private double[][] input;
        private double[][] output;

        public ConvolutionalLayer() {
            this.filter = new double[3][3];
            // Iniciar filtros
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    filter[i][j] = (RANDOM.nextDouble() - 0.5) * 0.01;
                }
            }
        }

        public double[][] forward(double[][] input) {
            this.input = input;
            int inputSize = input.length;
            int outputSize = inputSize - 3 + 1; // 5x5 -> 3x3 output
            this.output = new double[outputSize][outputSize];

            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double sum = 0.0;
                    for (int fi = 0; fi < 3; fi++) {
                        for (int fj = 0; fj < 3; fj++) {
                            sum += input[i + fi][j + fj] * filter[fi][fj];
                        }
                    }
                    output[i][j] = relu(sum + bias);
                }
            }
            return output;
        }

        public double[][] backward(double[][] outputGradient) {
            // 1. Apply ReLU derivative element-wise to the gradient
            double[][] activatedGradient = new double[output.length][output.length];
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output.length; j++) {
                    // Chain rule: dLoss/dOutput * dOutput/dNetInput
                    activatedGradient[i][j] = outputGradient[i][j] * reluDerivative(output[i][j]);
                }
            }

            double[][] filterGradient = new double[3][3];
            double biasGradient = 0.0;

            int outputSize = activatedGradient.length;

            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    biasGradient += activatedGradient[i][j];
                    for (int fi = 0; fi < 3; fi++) {
                        for (int fj = 0; fj < 3; fj++) {
                            filterGradient[fi][fj] += input[i + fi][j + fj] * activatedGradient[i][j];
                        }
                    }
                }
            }

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    filter[i][j] -= LEARNING_RATE * filterGradient[i][j];
                }
            }
            bias -= LEARNING_RATE * biasGradient;

            return new double[input.length][input.length];
        }
    }

    static class PoolingLayer {
        private double[][] input;
        private double[][] output;
        private int[][] maxIndices;

        public double[][] forward(double[][] input) {
            this.input = input;
            int inputSize = input.length; // e.g., 3
            int outputSize = inputSize / 2;

            if (inputSize == 3) {
                outputSize = 1;
            }

            this.output = new double[outputSize][outputSize];
            this.maxIndices = new int[outputSize * outputSize][2]; // Stores (row, col)

            int index = 0;
            for (int i = 0; i < inputSize; i += 2) {
                for (int j = 0; j < inputSize; j += 2) {
                    // Check bounds to ensure we don't go past 3x3 matrix
                    if (i >= outputSize * 2 || j >= outputSize * 2) continue;

                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxR = -1, maxC = -1;

                    // Iterate over the 2x2 window
                    for (int r = 0; r < 2 && (i + r) < inputSize; r++) {
                        for (int c = 0; c < 2 && (j + c) < inputSize; c++) {
                            if (input[i + r][j + c] > maxVal) {
                                maxVal = input[i + r][j + c];
                                maxR = i + r;
                                maxC = j + c;
                            }
                        }
                    }

                    if (outputSize > 0) {
                        int outR = i / 2;
                        int outC = j / 2;
                        this.output[outR][outC] = maxVal;
                        // Store max position for backprop
                        this.maxIndices[index][0] = maxR;
                        this.maxIndices[index][1] = maxC;
                        index++;
                    }
                }
            }

            return output;
        }

        public double[][] backward(double[][] outputGradient) {
            double[][] inputGradient = new double[input.length][input.length];
            int index = 0;
            for (int i = 0; i < outputGradient.length; i++) {
                for (int j = 0; j < outputGradient[i].length; j++) {
                    int maxR = maxIndices[index][0];
                    int maxC = maxIndices[index][1];

                    if (maxR != -1 && maxC != -1) {
                        inputGradient[maxR][maxC] = outputGradient[i][j];
                    }
                    index++;
                }
            }
            return inputGradient;
        }
    }

    static class DenseLayer {
        private final double[][] weights;
        private double[] bias;
        private double[] input;
        private double[] output;
        private final int inputSize;
        private final int outputSize;

        public DenseLayer(int inputSize, int outputSize) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = new double[inputSize][outputSize];
            this.bias = new double[outputSize];

            // Initialize weights and bias randomly
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights[i][j] = (RANDOM.nextDouble() - 0.5) * 0.01;
                }
            }
            for (int j = 0; j < outputSize; j++) {
                bias[j] = RANDOM.nextDouble() * 0.1;
            }
        }

        public double[] forward(double[] input) {
            this.input = input;
            this.output = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                double sum = 0.0;
                for (int i = 0; i < inputSize; i++) {
                    sum += input[i] * weights[i][j];
                }
                output[j] = sigmoid(sum + bias[j]);
            }
            return output;
        }

        public double[] backward(double[] outputGradient) {
            double[] inputGradient = new double[inputSize];
            double[][] weightGradient = new double[inputSize][outputSize];
            double[] biasGradient = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                double activatedGradient = outputGradient[j] * sigmoidDerivative(output[j]);
                biasGradient[j] = activatedGradient;

                for (int i = 0; i < inputSize; i++) {
                    weightGradient[i][j] = activatedGradient * input[i];

                    inputGradient[i] += activatedGradient * weights[i][j];
                }
            }

            for (int j = 0; j < outputSize; j++) {
                bias[j] -= LEARNING_RATE * biasGradient[j];
                for (int i = 0; i < inputSize; i++) {
                    weights[i][j] -= LEARNING_RATE * weightGradient[i][j];
                }
            }
            return inputGradient; // Gradient passed back to the previous layer
        }
    }


    public static Image CargarImagen(String nombre)
    {
        File file = new File(nombre);
        BufferedImage img;
        int width ;
        int height;
        double[] img_data;
        int[][] imgArr;
        Raster raster ;

        try {
            img     = ImageIO.read(file);
            width   = img.getWidth();
            height = img.getHeight();
            System.out.printf("imagen con ancho %s, y alto %s. Cargada.\n", width, height);
            imgArr = new int[width][height];
            raster = img.getData();

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                imgArr[i][j] = raster.getSample(i, j, 0);
            }
        }

        img_data = new double[width * height];

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                img_data[(i * height) + j] = ((double)(imgArr[i][j])) / 255.0;
            }
        }

        return new Image(width, height, img_data);

        } catch (Exception e) {
            System.out.printf("Error cargando imagen");
        }

        return null;

    }

    public static void main(String[] args)
    {

        System.out.println("Cargando Imagenes y convirtiendo a valores flotantes (double).\n");

        Image imagen1 = CargarImagen("ardilla_gris_160x120.jpg");
        Image imagen2 = CargarImagen("ave_gris_160x120.jpg");
        Image imagen4 = CargarImagen("loro_gris_160x120.jpg");
        Image imagen3 = CargarImagen("cachorro_gris_160x120.jpg");
        Image imagen5 = CargarImagen("gato_gris_160x120.jpg");

        double[][][] X_train1 = imagen1.data;
        double[] Y_train = { 1.0, 0.0, 1.0, 0.0, 1.0 }; // Target labels

        // 2. Initialize Layers
        ConvolutionalLayer convLayer = new ConvolutionalLayer(); // 5x5 -> 3x3
        PoolingLayer poolLayer = new PoolingLayer();             // 3x3 -> 1x1

        DenseLayer denseLayer = new DenseLayer(1, 1); // 1 entrada -> 1 salida

        System.out.println("--- Starting CNN Training (Gradient Descent) ---");
        System.out.printf("Epochs: %d, Learning Rate: %.4f\n", EPOCHS, LEARNING_RATE);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            double totalLoss = 0.0;
            int numCorrect = 0;

            for (int i = 0; i < X_train.length; i++) {
                double[][] image = X_train[i];
                double target = Y_train[i];

                double[][] convOutput = convLayer.forward(image);
                double[][] poolOutput = poolLayer.forward(convOutput);

                double[] denseInput = { poolOutput[0][0] };

                double[] prediction = denseLayer.forward(denseInput);
                double output = prediction[0];

                // calcular perdida o error
                totalLoss += mseLoss(output, target);

                if ((output >= 0.5 && target == 1.0) || (output < 0.5 && target == 0.0)) {
                    numCorrect++;
                }

                double outputError = mseLossDerivative(output, target);
                double[] outputGradient = { outputError };

                double[] denseInputGradient = denseLayer.backward(outputGradient);

                double[][] poolGradient = new double[1][1];
                poolGradient[0][0] = denseInputGradient[0]; // Gradient is simply passed back to the 1x1 pool output

                double[][] convGradient = poolLayer.backward(poolGradient);

                convLayer.backward(convGradient); // The gradient for the input image is ignored
            }

            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                double avgLoss = totalLoss / X_train.length;
                double accuracy = (double) numCorrect / X_train.length * 100;
                System.out.printf("Epoch %d: Loss = %.6f, Accuracy = %.1f%%\n", epoch + 1, avgLoss, accuracy);
            }
        }

        System.out.println("\n--- Final Test Predictions ---");
        for (int i = 0; i < X_train.length; i++) {
            double[][] image = X_train[i];
            double target = Y_train[i];

            // Forward Pass
            double[][] convOutput = convLayer.forward(image);
            double[][] poolOutput = poolLayer.forward(convOutput);
            double[] denseInput = { poolOutput[0][0] };
            double output = denseLayer.forward(denseInput)[0];

            String result = (output >= 0.5) ? "1 (Predicted: " : "0 (Predicted: ";
            System.out.printf("Example %d: Target: %.0f, %s%.4f)\n", i + 1, target, result, output);
        }
    }
}
