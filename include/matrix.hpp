#pragma once
#include <cstddef>
#include <string> 
#include <vector>

#define PAD_WIDTH 16

struct Matrix {
    int rows;
    int columns;
    int stride;
    int padded_size;
    std::vector<float> data;

    Matrix(int rows_, int columns_)
        : rows(rows_),
          columns(columns_),
          stride(((columns_ + PAD_WIDTH - 1) / PAD_WIDTH) * PAD_WIDTH),
          padded_size((size_t)rows * stride),
          data((size_t)rows_ * columns_, 0.0f)
    {}
};

size_t load_file(const std::string& filepath, const char*& file_data);
void unload_file(const char* file_data, size_t file_size);
Matrix load_one_matrix(const char*& ptr, const char* end);