// BSD 3-Clause License
//
// Copyright (c) 2021, 2022, Hans-Martin Will
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SLAMMER_OPTIMIZER_H
#define SLAMMER_OPTIMIZER_H

#pragma once

#include "slammer/slammer.h"

#include "Eigen/Sparse"

namespace slammer {

namespace sparse {

template <typename Matrix, typename Iterator>
void EmitTriplets(const Matrix& matrix, Iterator output, size_t row_offset, size_t column_offset) {
    for (size_t row_index = 0; row_index < matrix.rows(); ++row_index) {
        for (size_t column_index = 0; column_index < matrix.cols(); ++column_index) {
            auto value = matrix(row_index, column_index);

            if (value) {
                *output++ = Eigen::Triplet<double>(row_offset + row_index, column_offset + column_index, value);
            }
        }
    }
}

} // namespace sparse

/// The signature of the function to compute the Jacobian that is passed to the least squares optimizer
/// functions.
typedef std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd& value)> Jacobian;

/// The signature of the function to compute the Jacobian that is passed to the least squares optimizer
/// functions.
typedef std::function<Eigen::VectorXd(const Eigen::VectorXd& value)> Residual;

/// Least-squares optimization using Gauss-Newton iteration.
///
/// \param jacobian_function    function that can calculate the Jaconbian for a given set of parameters values.
/// \param residual_function    function that can calculate the residual vector for a given set of parameters values.
/// \param value                the value to be optimized. It serves as initial value and is updated
///                             as the optimization progresses.
/// \param max_iterations       Maximum number of iterations to perform
///
/// \returns        the squared error associated with the value at the end of the optimization procedure, 
///                 which is the sum of the squares of the residuals
Result<double> GaussNewton(const Jacobian& jacobian_function, const Residual& residual_function, 
                           Eigen::VectorXd& value, unsigned max_iterations);

/// Least-squares optimization using Levenberg-Marquardt iteration.
///
/// \param jacobian_function    function that can calculate the Jaconbian for a given set of parameters values.
/// \param residual_function    function that can calculate the residual vector for a given set of parameters values.
/// \param value                the value to be optimized. It serves as initial value and is updated
/// \param max_iterations        maximum number of iterations to perform
/// \param lambda               the lambda parameter, which determines blending between Gauss-Newton and gradient 
///                             descent
///
/// \returns        the squared error associated with the value at the end of the optimization procedure, 
///                 which is the sum of the squares of the residuals
Result<double> LevenbergMarquardt(const Jacobian& jacobian_function, const Residual& residual_function,
                                  Eigen::VectorXd& value, unsigned max_iterations, 
                                  double lambda);

} // namespace slammer

#endif //ndef SLAMMER_OPTIMIZER_H
