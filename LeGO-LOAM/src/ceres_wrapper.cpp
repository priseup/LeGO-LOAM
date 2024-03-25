#include "ceres_wrapper.h"

void solve_problem(ceres::Problem &problem)
{
    ceres::Solver::Summary summary;

    ceres::Solver::Options options;
    options.max_num_iterations = 5;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solve(options, &problem, &summary);
}solve_problem
