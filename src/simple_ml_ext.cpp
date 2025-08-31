#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float *Z_batch = new float[batch * k]; // [batch, k]
    float *G = new float[n * k]; // Gradient
    for(size_t x1 = 0; x1 < m; x1+=batch) {
        // Perform exp(X_batch @ theta), [m, n] @ [n, k] = [m, k]
        for(size_t b = 0; b < batch; b+=1) {
            for(size_t t2 = 0; t2 < k; t2++) {
                auto sum = 0.0f;
                for(size_t x2 = 0; x2 < n; x2++) {
                    sum += X[(x1 + b) * n + x2] * theta[x2 * k + t2];
                }
                Z_batch[b * k + t2] = exp(sum);
            }
        }
        // Normalize Z_batch row-wise, substract IY
        for(size_t z1 = 0; z1 < batch; z1++) {
            auto row_sum = 0.0f;
            for(size_t z2 = 0; z2 < k; z2++) {
                row_sum += Z_batch[z1 * k + z2];
            }
            for(size_t z2 = 0; z2 < k; z2++) {
                Z_batch[z1 * k + z2] = 
                    Z_batch[z1 * k + z2]/row_sum -
                    (y[x1 + z1] == z2? 1.0f : 0.0f);
            }
        }
        // Clean Gradient
        std::fill(G, G + n*k, 0);
        // Calculate Gradient = X_batch.T @ Z_batch [n, batch] @ [batch, k] = [n, k]
        // Make step after we have current element
        for(size_t x2 = 0; x2 < n; x2++) {
            for(size_t z2 = 0; z2 < k; z2++) {
                auto sum = 0.0f;
                for(size_t b = 0; b < batch; b++) {
                    sum += X[(x1 + b) * n + x2] * Z_batch[b * k + z2];
                }
                G[x2 * k + z2] = sum;
            }
        }
        // Update Theta
        for(size_t i = 0; i < n * k; i++) {
            theta[i] -= lr/static_cast<float>(batch) * G[i];
        }
    }
    // Clean
    delete[] Z_batch;
    delete[] G;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
