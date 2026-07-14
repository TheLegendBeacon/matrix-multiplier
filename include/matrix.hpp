#pragma once
#include <cstddef>
#include <ostream>
#include <string> 
#include <vector>

constexpr int pad_width(int width, int PAD_WIDTH) {
    return (width + PAD_WIDTH-1) / PAD_WIDTH * PAD_WIDTH;
}

constexpr int pad_height(int height, int PAD_HEIGHT) {
    return (height + PAD_HEIGHT-1) / PAD_HEIGHT * PAD_HEIGHT;
}

struct Matrix {
    int rows;
    int columns;
    alignas(32) std::vector<float> data;

    Matrix(int rows_, int columns_)
        : rows(rows_),
          columns(columns_),
          data((size_t)rows_ * columns_, 0.0f)
    {}

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        os << matrix.rows << ' ' << matrix.columns << std::endl;
        long long counter = 0;
        for (auto k: matrix.data) {
            os << k << ' ';
            counter++;
            if (counter % matrix.columns == 0) os << std::endl;
        }
        return os;
    }
};

size_t load_file(const std::string& filepath, const char*& file_data);
void unload_file(const char* file_data, size_t file_size);
Matrix load_one_matrix(const char*& ptr, const char* end);
Matrix multiply(Matrix a, Matrix b);