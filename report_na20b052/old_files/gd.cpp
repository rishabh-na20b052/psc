#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <bits/stdc++.h>
#include "input.h"
#include "mpi.h"

using namespace std;

void compute_gradient(double** x_local, double* y_local, double* theta, int n,int m, double* gradient) {
    
    double y_hat[n];
    // calculating the predicted values using the current theta values
    for(int i = 0; i < n; i++) {
        y_hat[i] = 0;
        for(int j = 0; j < m; j++) {
            y_hat[i] += x_local[i][j] * theta[j];
        }
    }

    double error[n];
    // calculating the gradient
    for(int i = 0; i < n; i++) {
        error[i] = y_hat[i] - y_local[i];
        for(int j = 0; j < m; j++) {
            gradient[j] += x_local[i][j] * error[i];
        }
    }
    for(int j = 0; j < m; j++) {
        gradient[j] /= n;
    }
}

int main(int argc, char **argv ) {
    
    int rank, size, tag=100;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 1960; // total number of data points ( needs to be changed according to the dataset)
    
    int n ; // local number of data points
    int start = (N*rank)/size; // start index of local data points  
    int end = (N*(rank+1))/size - 1; // end index of local data points
    n = end - start + 1;

    int f = 7; // number of features
    int d = 2; // degree of basis polynomial
    int m = tgamma(f+d+1)/(tgamma(f+1)*tgamma(d+1)); // number of terms in basis polynomial
    

    double** x_local = new double*[n];
    for (int i = 0; i < n; ++i)
        x_local[i] = new double[m];
    
    double x_local_recv[n][m]; // local data points
    double y_local[n]; // local data points

    // distribute data points to all processes
    if(rank == 0) {
        vector<vector<double>> x_total = read_x();
        vector<vector<double>> y_total = read_y();

        for(int i = 0; i < n; i++) {
            x_local[i][0] = 1; // bias
            for (int j = 1; j <= f; j++) {
                x_local[i][j] = x_total[start + i][j];
                y_local[i] = y_total[start + i][0];
            }
            int j = f+1;
            for(int k = 0; k<f; k++){
                for(int l = k; l<f; l++){
                    x_local[i][j] = x_total[start + i][k]*x_total[start + i][l];
                    j++;
                }
            }
        }

        for(int r = 1; r < size; r++) {
            int start_i = (N*r)/size;
            int end_i = (N*(r+1))/size - 1;
            int n_i = end_i - start_i + 1;
            double x_local_i[n_i][m];
            double y_local_i[n_i];
            for(int i = 0; i < n; i++) {
                x_local_i[i][0] = 1; // bias
                for (int j = 1; j <= f; j++) {
                    x_local_i[i][j] = x_total[start_i + i][j];
                    y_local_i[i] = y_total[start_i + i][0];
                }
                int j = f+1;
                for(int k = 0; k<f; k++){
                    for(int l = k; l<f; l++){
                        x_local_i[i][j] = x_total[start_i + i][k]*x_total[start_i + i][l];
                        j++;
                    }
                }
            }
            MPI_Send(&x_local_i, n_i*m, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            MPI_Send(&y_local_i, n_i, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
        }
        x_total.clear();
        y_total.clear();
    }
    else {
        MPI_Recv(&x_local_recv, n*m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&y_local, n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                x_local[i][j] = x_local_recv[i][j];
            }
        }

    } 

    // initialize parameters
    double theta[m]= {2}; // parameters
    double learning_rate = stod(argv[1]);
    int num_iterations = stoi(argv[2]);
    
    for(int iter = 0; iter < num_iterations; iter++) {
        double gradient[m] = {0};
        compute_gradient(x_local, y_local, theta, n, m, gradient);

        double* red_gradient = nullptr; // Declare red_gradient
        if(rank == 0)
            red_gradient = (double*)malloc(m*sizeof(double));
        MPI_Reduce(gradient, red_gradient, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // reduced_gradient is sum of gradients calculated over all the processes.
        // the real gradient vector will be given by red_gradient/size;
        

        // update theta values to move towards negative gradient direction
        if(rank == 0){
            for(int j = 0; j < m; j++) {
                theta[j] -= (learning_rate * red_gradient[j])/size;
            }
        }
        if(size > 1) MPI_Bcast(theta, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    
    
    cout<<"root mean sq error from rank "<< rank<<" : ";
    double sq_error = 0;
    for(int i = 0; i < n; i++) {
        double prediction = 0;
        for(int j = 0; j < m; j++) {
            prediction += x_local[i][j] * theta[j];
        }
        sq_error += (prediction - y_local[i]) * (prediction - y_local[i]);
    }
    cout<<sqrt(sq_error/n)<<endl;

    return 0;
}