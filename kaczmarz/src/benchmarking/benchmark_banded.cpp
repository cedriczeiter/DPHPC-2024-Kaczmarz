#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "../solvers/banded.hpp"

using hrclock = std::chrono::high_resolution_clock;

const auto solvers = []() {
  std::unordered_map<std::string, std::unique_ptr<BandedSolver>> solvers;
  /*
  solvers["openMP_grouping1_2_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(2);
  solvers["openMP_grouping1_3_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(3);
  solvers["openMP_grouping1_4_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(4);
  solvers["openMP_grouping1_5_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(5);
  solvers["openMP_grouping1_6_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(6);
  solvers["openMP_grouping1_8_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(8);
  solvers["openMP_grouping2_2_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(2);
  solvers["openMP_grouping2_3_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(3);
  solvers["openMP_grouping2_4_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(4);
  solvers["openMP_grouping2_5_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(5);
  solvers["openMP_grouping2_6_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(6);
  solvers["openMP_grouping2_8_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(8);
  */
  solvers["CUDA_grouping1_128_8"] = std::make_unique<CUDAGrouping1BandedSolver>(8, 128);
  solvers["CUDA_grouping1_128_16"] = std::make_unique<CUDAGrouping1BandedSolver>(16, 128);
  solvers["CUDA_grouping2_128_8"] = std::make_unique<CUDAGrouping2BandedSolver>(8, 128);
  solvers["CUDA_grouping2_128_16"] = std::make_unique<CUDAGrouping2BandedSolver>(16, 128);
  solvers["CUDA_grouping1_140_8"] = std::make_unique<CUDAGrouping1BandedSolver>(8, 140);
  solvers["CUDA_grouping1_140_16"] = std::make_unique<CUDAGrouping1BandedSolver>(16, 140);
  solvers["CUDA_grouping2_140_8"] = std::make_unique<CUDAGrouping2BandedSolver>(8, 140);
  solvers["CUDA_grouping2_140_16"] = std::make_unique<CUDAGrouping2BandedSolver>(16, 140);
  solvers["openMP_grouping1_2_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(2);
  solvers["openMP_grouping1_3_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(3);
  solvers["openMP_grouping1_4_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(4);
  solvers["openMP_grouping1_5_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(5);
  solvers["openMP_grouping1_6_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(6);
  solvers["openMP_grouping1_8_threads"] = std::make_unique<OpenMPGrouping1BandedSolver>(8);
  solvers["openMP_grouping2_2_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(2);
  solvers["openMP_grouping2_3_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(3);
  solvers["openMP_grouping2_4_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(4);
  solvers["openMP_grouping2_5_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(5);
  solvers["openMP_grouping2_6_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(6);
  solvers["openMP_grouping2_8_threads"] = std::make_unique<OpenMPGrouping2BandedSolver>(8);
  solvers["serial_naive"] = std::make_unique<SerialNaiveBandedSolver>();
  solvers["serial_interleaved"] = std::make_unique<SerialInterleavedBandedSolver>();
  return solvers;
}();

struct BandedBenchmarkTask {
  std::string solver_name;
  unsigned matrix_dim;
  unsigned matrix_bandwidth;
  unsigned iteration_count;
  bool double_precision;
  unsigned random_seed;
};

struct BandedBenchmarkResult {
  long long running_time_ns;
  double error_to_eigen_L1;
  double error_to_eigen_L2;
  double error_to_eigen_Linf;
  long long execution_timestamp; // when the benchmark was executed so that we can investigate possible running time deviations
};

static long long duration_ns(const hrclock::time_point t_start, const hrclock::time_point t_end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
}

static BandedBenchmarkResult benchmark(const BandedBenchmarkTask& task) {
  std::mt19937 rng(task.random_seed);
  const BandedLinearSystem lse =
      BandedLinearSystem::generate_random_regular(rng, task.matrix_dim, task.matrix_bandwidth);

  const Vector x_eigen = lse.to_sparse_system().eigen_solve();

  Vector x_kaczmarz = Vector::Zero(task.matrix_dim);

  BandedSolver *const solver = solvers.at(task.solver_name).get();

  BandedBenchmarkResult result;

  result.execution_timestamp = time(NULL);

  const auto kaczmarz_start = hrclock::now();
  solver->run_iterations(lse, x_kaczmarz, task.iteration_count);
  const auto kaczmarz_end = hrclock::now();

  const Vector x_error = x_kaczmarz - x_eigen;

  result.running_time_ns = duration_ns(kaczmarz_start, kaczmarz_end);
  result.error_to_eigen_L1 = x_error.lpNorm<1>();
  result.error_to_eigen_L2 = x_error.lpNorm<2>();
  result.error_to_eigen_Linf = x_error.lpNorm<Eigen::Infinity>();

  return result;
}

std::vector<BandedBenchmarkTask> generate_benchmark_tasks() {
  constexpr unsigned repetitions = 3;
  std::vector<BandedBenchmarkTask> tasks;
  //const std::vector<unsigned> matrix_dims = { 100, 200, 500, 1000, 2000, 5000, 10'000, 20'000, 50'000 };
  //const std::vector<unsigned> matrix_dims = { 10'000, 20'000, 40'000, 60'000, 80'000, 100'000 };
  for (unsigned rep = 0; rep < repetitions; rep++) {
    for (const auto& solver : solvers) {
      //for (const unsigned matrix_dim : matrix_dims) {
        tasks.emplace_back(solver.first, 100'000, 2, 10'000, true, 21);
      //}
    }
  }
  return tasks;
}

int main() {
  const auto tasks = []() {
    std::mt19937 rng(0);
    std::vector<BandedBenchmarkTask> tasks = generate_benchmark_tasks();
    std::shuffle(tasks.begin(), tasks.end(), rng);
    return tasks;
  }();

  const std::string outfile = "banded_benchmarks.csv";

  std::cout << "will write to file " << outfile << std::endl;

  std::ofstream ofs(outfile);
  const auto print_benchmark_result = [&ofs](const BandedBenchmarkTask& task, const BandedBenchmarkResult& result) {
    ofs << task.solver_name << "," << task.matrix_dim << "," << task.matrix_bandwidth << "," << task.iteration_count << "," << task.double_precision << "," << task.random_seed << "," << result.running_time_ns << "," << result.error_to_eigen_L1 << "," << result.error_to_eigen_L2 << "," << result.error_to_eigen_Linf << "," << result.execution_timestamp << std::endl;
  };

  ofs << "solver_name,matrix_dim,matrix_bandwidth,iteration_count,double_precision,random_seed,running_time_ns,error_to_eigen_L1,error_to_eigen_L2,error_to_eigen_Linf,execution_timestamp\n";

  for (size_t i = 0; i < tasks.size(); i++) {
    std::cout << "running task " << i + 1 << " out of " << tasks.size() << std::endl;
    print_benchmark_result(tasks[i], benchmark(tasks[i]));
  }
}

