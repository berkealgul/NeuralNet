using System;

namespace NeuralNetworkLib
{
    public class Matrix
    {
        public int rowCount;
        public int columnCount;

        public float[,] values; //actual matrix

        public Matrix(int rowCount, int columnCount)
        {
            if (rowCount < 1 || columnCount < 1)
                throw new ArgumentException("Matrises row or column count cant less than 1. Check constructor parameters");

            this.columnCount = columnCount;
            this.rowCount = rowCount;

            values = new float[rowCount, columnCount];
        }

        public Matrix(int rowCount, int columnCount, float[,] values)
        {
            this.columnCount = columnCount;
            this.rowCount = rowCount;
            this.values = values;
        }

        public void multiply(float scaler)
        {
            for (int j = 0; j < rowCount; j++)
            {
                for (int i = 0; i < columnCount; i++)
                {
                    values[j, i] *= scaler;
                }
            }
        }


        public void multiply(Matrix otherMatrix)
        {
            if (columnCount != otherMatrix.columnCount || rowCount != otherMatrix.rowCount)
                throw new InvalidOperationException("'otherMatrix' must have same dimensions with source matrix");

            for (int j = 0; j < rowCount; j++)
            {
                for (int i = 0; i < columnCount; i++)
                {
                    values[j, i] *= otherMatrix.values[j, i];
                }
            }
        }


        public void add(Matrix otherMatrix)
        {
            if (columnCount != otherMatrix.columnCount || rowCount != otherMatrix.rowCount)
                throw new InvalidOperationException("'otherMatrix' must have same dimensions with source matrix");

            for (int j = 0; j < rowCount; j++)
            {
                for (int i = 0; i < columnCount; i++)
                {
                    values[j, i] += otherMatrix.values[j, i];
                }
            }
        }

        public void add(float number)
        {
            for (int j = 0; j < rowCount; j++)
            {
                for (int i = 0; i < columnCount; i++)
                {
                    values[j, i] += number;
                }
            }
        }


        public void sub(Matrix otherMatrix)
        {
            otherMatrix.multiply(-1);
            add(otherMatrix);
        }

        public void sub(float num)
        {
            add(-num);
        }


        public Matrix transpose() // T(3x2) = 2x3
        {
            Matrix newMatrix = new Matrix(columnCount, rowCount);

            for (int j = 0; j < newMatrix.rowCount; j++)
            {
                for (int i = 0; i < newMatrix.columnCount; i++)
                {
                    newMatrix.values[j, i] = values[i, j];
                }
            }

            return newMatrix;
        }


        public static Matrix Product(Matrix firstMatrix, Matrix secMatrix) // m x n * n x a = m x a 
        {
            if (firstMatrix.columnCount != secMatrix.rowCount)
                throw new InvalidOperationException("'firstMatrix's column count must equals 'secMatrix's row count");

            Matrix newMatrix = new Matrix(firstMatrix.rowCount, secMatrix.columnCount);

            for (int j = 0; j < newMatrix.rowCount; j++)
            {
                for (int i = 0; i < newMatrix.columnCount; i++)
                {
                    float[] row = new float[firstMatrix.columnCount];
                    float[] column = new float[secMatrix.rowCount];

                    for (int x = 0; x < row.Length; x++)
                    {
                        row[x] = firstMatrix.values[j, x];
                        column[x] = secMatrix.values[x, i];
                    }

                    newMatrix.values[j, i] = dotproduct(row, column);
                }
            }

            return newMatrix;
        }


        public Matrix copy()
        {
            Matrix copy = new Matrix(rowCount, columnCount);

            for (int i = 0; i < columnCount; i++)
            {
                for (int j = 0; j < rowCount; j++)
                {
                    copy.values[j, i] = values[j, i];
                }
            }

            return copy;
        }


        private static float dotproduct(float[] firstA, float[] secA)
        {
            if (firstA.Length != secA.Length)
                throw new InvalidOperationException("arrays must have same lenght");

            float result = 0;

            for (int i = 0; i < firstA.Length; i++)
            {
                result += firstA[i] * secA[i];
            }

            return result;
        }
    }
}