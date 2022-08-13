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

#include "slammer/optimizer.h"

#include "Eigen/SparseCholesky"

using namespace slammer;

Result<double> 
slammer::GaussNewton(const Jacobian& jacobian_function, const Residual& residual_function,
                     Eigen::VectorXd& value, unsigned max_iterations) {
    while (max_iterations--) {
        auto jacobian = jacobian_function(value);
        auto residual = residual_function(value);
        auto error = residual.squaredNorm();

        // std::cout << "Value = " << value.transpose() << std::endl;
        // std::cout << "Residual = " << residual.transpose() << std::endl;
        // std::cout << "Squared error =  " << error << std::endl;

        Eigen::SparseMatrix<double> lhs = jacobian.transpose() * jacobian;
        Eigen::VectorXd rhs = jacobian.transpose() * residual;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
        solver.compute(lhs);

        if(solver.info() != Eigen::Success) {
            return Error("slammer::GaussNewton: Failed to factorize matrix");
        }

        Eigen::VectorXd delta = solver.solve(rhs);

        if(solver.info() != Eigen::Success) {
            return Error("slammer::GaussNewton: Failed to solve for right-hand side");
        }

        // std::cout << "Delta = " << delta.transpose() << std::endl;
        value -= delta;
    }

    auto residual = residual_function(value);
    return residual.squaredNorm();
}

Result<double> 
slammer::LevenbergMarquardt(const Jacobian& jacobian_function, const Residual& residual_function,
                            Eigen::VectorXd& value,
                            unsigned max_iterations, double lambda) {    
    auto residual = residual_function(value);
    auto error = residual.squaredNorm();
    auto jacobian = jacobian_function(value);

    while (max_iterations--) {
        auto dim = jacobian.cols();
        Eigen::SparseMatrix<double> lambda_identity(dim, dim);
        lambda_identity.setIdentity();
        lambda_identity *= lambda;

        // std::cout << "Value = " << value.transpose() << std::endl;
        // std::cout << "Residual = " << residual.transpose() << std::endl;
        // std::cout << "Squared error =  " << error << std::endl;

        Eigen::SparseMatrix<double> lhs = jacobian.transpose() * jacobian + lambda_identity;
        Eigen::VectorXd rhs = jacobian.transpose() * residual;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
        solver.compute(lhs);

        if(solver.info() != Eigen::Success) {
            return Error("slammer::LevenbergMarquardt: Failed to factorize matrix");
        }

        Eigen::VectorXd delta = solver.solve(rhs);

        if(solver.info() != Eigen::Success) {
            return Error("slammer::LevenbergMarquardt: Failed to solve for right-hand side");
        }

        // std::cout << "Delta = " << delta.transpose() << std::endl;
        auto candidate_value = value - delta;
        auto candidate_residual = residual_function(candidate_value);
        auto candidate_error = candidate_residual.squaredNorm();
        // std::cout << "Candidate error =  " << candidate_error << std::endl;

        if (candidate_error < error) {
            value = candidate_value;
            residual = candidate_residual;
            error = candidate_error;
            jacobian = jacobian_function(value);
            lambda *= 0.1;
        } else {
            lambda *= 10;
        }
    }

    return error;
}
