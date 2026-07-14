#include "matrix.hpp"
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc == 1) {
        std::cout << "Usage: ./matrix_app <path-to-file>" << std::endl;
        return 1;
    }
    std::string path = argv[1];

    const char* fileptr;
    size_t file_size = load_file(path, fileptr);

    const char* start = fileptr;

    if (file_size == 0) {
        std::cout << "Error Loading file: " << path << std::endl;
        return 1;
    }

    Matrix matrix1 = load_one_matrix(fileptr, start+file_size);
    Matrix matrix2 = load_one_matrix(fileptr, start+file_size);
    unload_file(start, file_size);

    auto t0 = std::chrono::steady_clock::now();
    Matrix result = multiply(matrix1, matrix2);
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "multiply took " << elapsed_us << " us\n";
    
    double total_operations = 2.0 * result.rows * matrix1.columns * result.columns;
    double gflops = total_operations / std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    std::cout << "GFLOPS: " << gflops << std::endl;
    // std::cout << result;

    return 0;
}