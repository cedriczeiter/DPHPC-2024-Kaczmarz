#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "../solvers/nolpde.hpp"

using hrclock = std::chrono::high_resolution_clock;

const auto solvers = []() {
  std::unordered_map<std::string, std::unique_ptr<NolPDESolver>> solvers;
  /*
  solvers["openMP_grouping1_2_threads"] =
  std::make_unique<OpenMPGrouping1BandedSolver>(2);
  solvers["openMP_grouping1_3_threads"] =
  std::make_unique<OpenMPGrouping1BandedSolver>(3);
  solvers["openMP_grouping1_4_threads"] =
  std::make_unique<OpenMPGrouping1BandedSolver>(4);
  solvers["openMP_grouping1_5_threads"] =
  std::make_unique<OpenMPGrouping1BandedSolver>(5);
  solvers["openMP_grouping1_6_threads"] =
  std::make_unique<OpenMPGrouping1BandedSolver>(6);
  solvers["openMP_grouping1_8_threads"] =
  std::make_unique<OpenMPGrouping1BandedSolver>(8);
  solvers["openMP_grouping2_2_threads"] =
  std::make_unique<OpenMPGrouping2BandedSolver>(2);
  solvers["openMP_grouping2_3_threads"] =
  std::make_unique<OpenMPGrouping2BandedSolver>(3);
  solvers["openMP_grouping2_4_threads"] =
  std::make_unique<OpenMPGrouping2BandedSolver>(4);
  solvers["openMP_grouping2_5_threads"] =
  std::make_unique<OpenMPGrouping2BandedSolver>(5);
  solvers["openMP_grouping2_6_threads"] =
  std::make_unique<OpenMPGrouping2BandedSolver>(6);
  solvers["openMP_grouping2_8_threads"] =
  std::make_unique<OpenMPGrouping2BandedSolver>(8);
  */
  solvers["CUDA_128_16"] =
      std::make_unique<CUDANolPDESolver>(16, 128);
  solvers["basic_serial"] =
      std::make_unique<BasicSerialNolPDESolver>();
  solvers["permuting_serial"] =
      std::make_unique<PermutingSerialNolPDESolver>(16 * 128);
  solvers["shuffle_serial"] =
      std::make_unique<ShuffleSerialNolPDESolver>(292);
  return solvers;
}();

struct NolPDEBenchmarkTask {
  std::string solver_name;
  unsigned iteration_count;
  bool double_precision;
};

struct NolPDEBenchmarkResult {
  long long running_time_ns;
  double residual_L1;
  double residual_L2;
  double residual_Linf;
  long long
      execution_timestamp;  // when the benchmark was executed so that we can
                            // investigate possible running time deviations
};

static long long duration_ns(const hrclock::time_point t_start,
                             const hrclock::time_point t_end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
      .count();
}

const Discretization d = []() {
  std::ifstream ifs(
      "../../generated_bvp_matrices/problem3/"
      "problem3_complexity8_degree1.txt");
  return Discretization::read_from_stream(ifs);
}();

static NolPDEBenchmarkResult benchmark(const NolPDEBenchmarkTask& task) {

  Vector x_kaczmarz = Vector::Zero(d.sys.A().cols());

  NolPDESolver* const solver = solvers.at(task.solver_name).get();

  NolPDEBenchmarkResult result;

  result.execution_timestamp = time(NULL);

  const auto kaczmarz_start = hrclock::now();
  solver->run_iterations(d, x_kaczmarz, task.iteration_count);
  const auto kaczmarz_end = hrclock::now();

  const Vector residual = d.sys.b() - d.sys.A() * x_kaczmarz;

  result.running_time_ns = duration_ns(kaczmarz_start, kaczmarz_end);
  result.residual_L1 = residual.lpNorm<1>();
  result.residual_L2 = residual.lpNorm<2>();
  result.residual_Linf = residual.lpNorm<Eigen::Infinity>();

  return result;
}

std::vector<NolPDEBenchmarkTask> generate_benchmark_tasks() {
  constexpr unsigned repetitions = 5;
  std::vector<NolPDEBenchmarkTask> tasks;
  // const std::vector<unsigned> matrix_dims = { 100, 200, 500, 1000, 2000,
  // 5000, 10'000, 20'000, 50'000 }; const std::vector<unsigned> matrix_dims = {
  // 10'000, 20'000, 40'000, 60'000, 80'000, 100'000 };
  for (unsigned rep = 0; rep < repetitions; rep++) {
    for (const auto& solver : solvers) {
      // for (const unsigned matrix_dim : matrix_dims) {
      tasks.emplace_back(solver.first, 10'000, true);
      //}
    }
  }
  return tasks;
}

int main() {
  const auto tasks = []() {
    std::mt19937 rng(0);
    std::vector<NolPDEBenchmarkTask> tasks = generate_benchmark_tasks();
    std::shuffle(tasks.begin(), tasks.end(), rng);
    return tasks;
  }();

  const std::string outfile = "nolpde_benchmarks.csv";

  std::cout << "will write to file " << outfile << std::endl;

  std::ofstream ofs(outfile);
  const auto print_benchmark_result = [&ofs](
                                          const NolPDEBenchmarkTask& task,
                                          const NolPDEBenchmarkResult& result) {
    ofs << task.solver_name << ","
        << task.iteration_count << ","
        << task.double_precision << ","
        << result.running_time_ns << ","
        << std::format("{},{},{}", result.residual_L1,
                       result.residual_L2, result.residual_Linf)
        << "," << result.execution_timestamp << std::endl;
  };

  ofs << "solver_name,iteration_count,double_"
         "precision,running_time_ns,residual_L1,residual_L2,"
         "residual_Linf,execution_timestamp\n";

  for (size_t i = 0; i < tasks.size(); i++) {
    std::cout << "running task " << i + 1 << " out of " << tasks.size()
              << std::endl;
    print_benchmark_result(tasks[i], benchmark(tasks[i]));
  }
}
