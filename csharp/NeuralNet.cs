using System;

namespace NeuralNetworkLib
{
    public class NeuralNet
    {
        Random random = new Random();

        Matrix[] weights;
        Matrix[] biases;

        Matrix[] outputs_for_train;
        Matrix[] inputs_for_train;

        public NeuralNet(int input_neuron_count, int output_neuron_count, int hidden_neuron_count, int hiddenlayer_count)
        {
            weights = new Matrix[1 + hiddenlayer_count];
            biases = new Matrix[1 + hiddenlayer_count];
            outputs_for_train = new Matrix[1 + hiddenlayer_count];
            inputs_for_train = new Matrix[1 + hiddenlayer_count];

            //weight = matrix(next_layer_neuron_count x current_layer_neuron_count)
            //bias = matrix(next_layer_neuron_count x 1)

            weights[0] = new Matrix(hidden_neuron_count, input_neuron_count);
            biases[0] = new Matrix(hidden_neuron_count, 1);

            for (int i = 1; i < weights.Length - 1; i++)
            {
                weights[i] = new Matrix(hidden_neuron_count, hidden_neuron_count);
                biases[i] = new Matrix(hidden_neuron_count, 1);
            }

            weights[weights.Length - 1] = new Matrix(output_neuron_count, hidden_neuron_count);
            biases[weights.Length - 1] = new Matrix(output_neuron_count, 1);

            for (int i = 0; i < weights.Length; i++)
            {
                randomize(weights[i]);
                randomize(biases[i]);
            }
        }


        // jsondan yüklenecekse
        public NeuralNet(string jsonString)
        {

            JsonNN nnjson = JsonNN.FromJson(jsonString);

            int layerC = nnjson.Weights.Length;

            Matrix[] weights = new Matrix[layerC];
            Matrix[] biases = new Matrix[layerC];

            for (int i = 0; i < layerC; i++)
            {
                weights[i] = JsonLoader.loadMatrix(nnjson.Weights[i]);
                biases[i] = JsonLoader.loadMatrix(nnjson.Biases[i]);
            }

            this.weights = weights;
            this.biases = biases;
            this.outputs_for_train = new Matrix[layerC];
            this.inputs_for_train = new Matrix[layerC];
        }


        public void gradient_decent_train(Matrix error, float learning_rate)
        {
            //dw = 2 * lr * E * (1 - O) * O x I_T
            //db = 2 * lr * E * (1 - O) * O 

            for (int i = weights.Length - 1; i > -1; i--)
            {
                Matrix O = outputs_for_train[i];
                Matrix I = inputs_for_train[i];
                Matrix E = error.copy();

                E.multiply(2 * learning_rate);
                E.multiply(O);
                O.multiply(-1);
                O.add(1);
                E.multiply(O);

                Matrix I_T = I.transpose();

                biases[i].add(E);
                E = Matrix.Product(E, I_T);
                weights[i].add(E);

                Matrix W_T = weights[i].transpose();
                error = Matrix.Product(W_T, error);
            }
        }


        public Matrix Feedforward(Matrix input)
        {
            Matrix guess = input;

            for (int i = 0; i < weights.Length; i++)
            {
                inputs_for_train[i] = guess;

                guess = Matrix.Product(weights[i], guess);
                guess.add(biases[i]);
                Activate(guess);

                outputs_for_train[i] = guess;
            }

            return guess;
        }


        void Activate(Matrix matrix) //sigmoid function is used here. Some other activation functions may required later on.
        {
            for (int j = 0, n = matrix.rowCount; j < n; j++)
            {
                for (int i = 0, m = matrix.columnCount; i < m; i++)
                {
                    matrix.values[j, i] = 1 / (1 + (float)Math.Exp(-matrix.values[j, i]));
                }
            }
        }


        void randomize(Matrix matrix)
        {
            for (int j = 0, n = matrix.rowCount; j < n; j++)
            {
                for (int i = 0, m = matrix.columnCount; i < m; i++)
                {
                    matrix.values[j, i] = (float)random.NextDouble();
                }
            }
        }
    }
}
