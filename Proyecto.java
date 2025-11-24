
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

    // --- Layer Implementations ---

    /**
     *
     * Represents the Convolutional Layer.
     * Simplification: Uses a single 3x3 filter and zero-padding is omitted.
     */
    static class ConvolutionalLayer {
        private final double[][] filter; // 3x3 Filter (Weight)
        private double bias = RANDOM.nextDouble() * 0.1;
        private double[][] input;
        private double[][] output;

        public ConvolutionalLayer() {
            this.filter = new double[3][3];
            // Initialize filter weights randomly
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    filter[i][j] = (RANDOM.nextDouble() - 0.5) * 0.01;
                }
            }
        }

        /** Forward Pass: Convolution operation (simplified). */
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

        /** Backward Pass: Calculates gradients for weights and input. */
        public double[][] backward(double[][] outputGradient) {
            // 1. Apply ReLU derivative element-wise to the gradient
            double[][] activatedGradient = new double[output.length][output.length];
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output.length; j++) {
                    // Chain rule: dLoss/dOutput * dOutput/dNetInput
                    activatedGradient[i][j] = outputGradient[i][j] * reluDerivative(output[i][j]);
                }
            }

            // 2. Calculate Filter Gradient (dLoss/dFilter)
            double[][] filterGradient = new double[3][3];
            double biasGradient = 0.0;

            int outputSize = activatedGradient.length;

            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    biasGradient += activatedGradient[i][j];
                    for (int fi = 0; fi < 3; fi++) {
                        for (int fj = 0; fj < 3; fj++) {
                            // The input window * activatedGradient[i][j]
                            filterGradient[fi][fj] += input[i + fi][j + fj] * activatedGradient[i][j];
                        }
                    }
                }
            }

            // 3. Apply Gradient Descent (Update Weights and Bias)
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    filter[i][j] -= LEARNING_RATE * filterGradient[i][j];
                }
            }
            bias -= LEARNING_RATE * biasGradient;

            // 4. Calculate Input Gradient (dLoss/dInput) for previous layer (unnecessary here, but conceptually important)
            // This would involve a full convolution/transposed convolution with the flipped filter.
            // Simplified: return an empty gradient since this is the first layer.
            return new double[input.length][input.length];
        }
    }

    /**
     * [Image of Max Pooling Operation]
     * Represents the Max Pooling Layer (2x2 filter, stride 2).
     * Only implements Max Pooling. No weights to train.
     */
    static class PoolingLayer {
        private double[][] input;
        private double[][] output;
        private int[][] maxIndices; // Store indices of the max value for backprop

        /** Forward Pass: Max Pooling (2x2, stride 2). */
        public double[][] forward(double[][] input) {
            this.input = input;
            int inputSize = input.length; // e.g., 3
            int outputSize = inputSize / 2; // e.g., 1 (3/2 = 1 due to integer division)

            // Since 3x3 input -> 1x1 output (Max Pool 2x2, stride 2, no padding)
            if (inputSize == 3) {
                 // Simplification: In a 3x3 to 1x1 pool, the output is just the max of the whole matrix
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

        /** Backward Pass: Distributes gradient only to the max-activated neuron. */
        public double[][] backward(double[][] outputGradient) {
            double[][] inputGradient = new double[input.length][input.length];
            int index = 0;
            for (int i = 0; i < outputGradient.length; i++) {
                for (int j = 0; j < outputGradient[i].length; j++) {
                    // Gradient is passed only to the position that was the max in the window
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

    /**
     * Represents the Dense (Fully Connected) Layer.
     * This layer is placed after the pooling layer, connecting the features
     * to the final output.
     */
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

        /** Forward Pass: Matrix multiplication + bias + activation. */
        public double[] forward(double[] input) {
            this.input = input;
            this.output = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                double sum = 0.0;
                for (int i = 0; i < inputSize; i++) {
                    sum += input[i] * weights[i][j];
                }
                // Sigmoid activation for the output layer
                output[j] = sigmoid(sum + bias[j]);
            }
            return output;
        }

        /** Backward Pass: Calculates gradients and updates weights/bias using Gradient Descent. */
        public double[] backward(double[] outputGradient) {
            double[] inputGradient = new double[inputSize];
            double[][] weightGradient = new double[inputSize][outputSize];
            double[] biasGradient = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                // Chain rule: dLoss/dOutput * dOutput/dNetInput
                double activatedGradient = outputGradient[j] * sigmoidDerivative(output[j]);
                biasGradient[j] = activatedGradient;

                for (int i = 0; i < inputSize; i++) {
                    // dLoss/dWeight = activatedGradient * input[i]
                    weightGradient[i][j] = activatedGradient * input[i];

                    // dLoss/dInput (for previous layer)
                    inputGradient[i] += activatedGradient * weights[i][j];
                }
            }

            // Apply Gradient Descent (Update Weights and Bias)
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

        // 1. Mock Dataset: 5 different 5x5 images (grayscale) and their target labels (binary classification)
        double[][][] X_train1 = imagen1.data;
        double[] Y_train = { 1.0, 0.0, 1.0, 0.0, 1.0 }; // Target labels

        // 2. Initialize Layers
        ConvolutionalLayer convLayer = new ConvolutionalLayer(); // 5x5 -> 3x3
        PoolingLayer poolLayer = new PoolingLayer();             // 3x3 -> 1x1 (flattened to 1 neuron)

        // Flattening: The 1x1 output of the pooling layer becomes the input for the dense layer (1 feature)
        DenseLayer denseLayer = new DenseLayer(1, 1); // 1 Input (feature) -> 1 Output (prediction)

        System.out.println("--- Starting CNN Training (Gradient Descent) ---");
        System.out.printf("Epochs: %d, Learning Rate: %.4f\n", EPOCHS, LEARNING_RATE);

        // 3. Training Loop (Gradient Descent)
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            double totalLoss = 0.0;
            int numCorrect = 0;

            for (int i = 0; i < X_train.length; i++) {
                double[][] image = X_train[i];
                double target = Y_train[i];

                // --- Forward Pass (Prediction) ---
                // Conv Layer -> ReLU
                double[][] convOutput = convLayer.forward(image);
                // Pool Layer (Flattening occurs implicitly as 1x1 output is used)
                double[][] poolOutput = poolLayer.forward(convOutput);

                // Flatten the 1x1 pooled output into a single array for the Dense layer
                double[] denseInput = { poolOutput[0][0] };

                // Dense Layer -> Sigmoid -> Output
                double[] prediction = denseLayer.forward(denseInput);
                double output = prediction[0];

                // Loss calculation
                totalLoss += mseLoss(output, target);

                // Check accuracy
                if ((output >= 0.5 && target == 1.0) || (output < 0.5 && target == 0.0)) {
                    numCorrect++;
                }

                // --- Backward Pass (Backpropagation / Gradient Calculation) ---

                // Output Gradient: dLoss/dOutput
                double outputError = mseLossDerivative(output, target);
                double[] outputGradient = { outputError };

                // 1. Dense Layer Backprop (Updates Dense weights/bias)
                double[] denseInputGradient = denseLayer.backward(outputGradient);

                // 2. Un-flatten and apply Max Pool Backprop
                double[][] poolGradient = new double[1][1];
                poolGradient[0][0] = denseInputGradient[0]; // Gradient is simply passed back to the 1x1 pool output

                // 3. Max Pooling Layer Backprop (Distributes gradient to the correct pixel)
                double[][] convGradient = poolLayer.backward(poolGradient);

                // 4. Convolutional Layer Backprop (Updates Conv weights/bias)
                convLayer.backward(convGradient); // The gradient for the input image is ignored
            }

            // 4. Reporting
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                double avgLoss = totalLoss / X_train.length;
                double accuracy = (double) numCorrect / X_train.length * 100;
                System.out.printf("Epoch %d: Loss = %.6f, Accuracy = %.1f%%\n", epoch + 1, avgLoss, accuracy);
            }
        }

        // 5. Final Test
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
            System.out.printf("Example %d: Target: %.0f, %s%.4f)\n",
                    i + 1, target, result, output);
        }
    }
}
