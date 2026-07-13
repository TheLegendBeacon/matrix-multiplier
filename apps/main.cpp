#include "matrix.hpp"
#include <cstddef>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc == 1) {
        std::cout << "Usage: ./matrix_app <path-to-file>" << std::endl;
    }
    std::string path = argv[1];

    const char* fileptr;
    size_t file_size = load_file(path, fileptr);

    const char* start = fileptr;

    if (file_size == 0) {
        std::cout << "Error Loading file: " << path << std::endl;
        return 1;
    }

    Matrix matrix = load_one_matrix(fileptr, fileptr+file_size);
    unload_file(start, file_size);

    std::cout << matrix.rows << ' ' << matrix.columns << std::endl;
    long long counter = 0;
    for (auto k: matrix.data) {
        std::cout << k << ' ';
        counter++;
        if (counter % matrix.rows == 0) std::cout << std::endl;
    } 
    return 0;
}