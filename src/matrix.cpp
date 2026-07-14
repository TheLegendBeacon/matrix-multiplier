#include "matrix.hpp"

#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <mm_malloc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <immintrin.h>

using FileInformation = struct stat;

size_t load_file(const std::string& filepath, const char*& file_data) {
    int stream_index = open(filepath.c_str(), O_RDONLY);
    if (stream_index == -1) {
        return 0;
    }

    FileInformation file_info; 
    if (fstat(stream_index, &file_info) == -1) {
        close(stream_index);
        return 0;
    }

    size_t file_size = file_info.st_size;
    if (file_size == 0) {
        close(stream_index);
        return 0;
    }

    file_data = static_cast<const char*>(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, stream_index, 0)
    );

    if (file_data == MAP_FAILED) {
        close(stream_index);
        return 0;
    }
    
    close(stream_index);
    return file_size;
}

void unload_file(const char* file_data, size_t file_size) {
    if (file_size == 0) {
        return;
    }
    munmap(const_cast<char*>(file_data), file_size);
}

unsigned long long parse_next_uint64(unsigned long long init_number, const char*& ptr, const char* end) {
    unsigned long long number = init_number;
    const char* p = ptr;

    while (p < end && (*p <= ' ')) {
        p++;
    }

    if (p >= end) {
        std::fprintf(stderr, "parse_next_uint64: unexpected EOF\n");
        std::abort();
    }

    while (p < end && (*p >= '0' && *p <= '9')) {
        number = number*10 + (*p - '0');
        p++;
    }

    ptr = p;
    return number;
}

unsigned long long parse_next_uint64(const char*& ptr, const char* end) {
    return parse_next_uint64(0, ptr, end);
}

float parse_next_float(const char*& ptr, const char* end) {
    unsigned long long number = 0;
    const char* p = ptr;
    int sign = 1;
    int exp = 0;

    while (p < end && (*p <= ' ')) {
        p++;
    }

    if (p < end && (*p == '-' || *p == '+')) {
        sign = -1 + 2* (*p == '+');
        p++;
    }
    
    number = parse_next_uint64(number, p, end);
    if (p < end && *p == '.') {
        p++;
        const char* start_ptr = p;
        number = parse_next_uint64(number, p, end);
        exp -= static_cast<int>(p-start_ptr);
    }

    if (p < end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < end && (*p == '-')) {
            p++;
            exp -= parse_next_uint64(0, p, end);
        } else {
            if (p < end && (*p == '+')) p++;
            exp += parse_next_uint64(0, p, end);
        }
    }

    ptr = p;
    return static_cast<float>(number) * pow(10, exp) * sign;
}

Matrix load_one_matrix(const char*& ptr, const char* end) {
    const char* p = ptr;
    int rows = parse_next_uint64(0, p, end);
    int columns = parse_next_uint64(0, p, end);

    Matrix m(rows, columns);
    for (int i = 0; i < rows; i++) {
        float* row = m.data.data() + (size_t)i * m.columns;
        for (int j = 0; j < columns; j++) {
            row[j] = parse_next_float(p, end);
        }
    }

    ptr = p;
    return m;
}

inline void micro_kernel3x24_unsafe(float* a_stream, float* b_stream, Matrix& c, int K, int ii, int jj) {
    __m256 c11 = _mm256_setzero_ps();
    __m256 c12 = _mm256_setzero_ps();
    __m256 c13 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c22 = _mm256_setzero_ps();
    __m256 c23 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c32 = _mm256_setzero_ps();
    __m256 c33 = _mm256_setzero_ps();

    #pragma GCC unroll 4
    for (int k = 0; k < K; k++) {
        __m256 b1 = _mm256_loadu_ps(b_stream);
        __m256 b2 = _mm256_loadu_ps(b_stream + 8);
        __m256 b3 = _mm256_loadu_ps(b_stream + 16);

        __m256 a1 = _mm256_broadcast_ss(a_stream);
        __m256 a2 = _mm256_broadcast_ss(a_stream + 1);
        __m256 a3 = _mm256_broadcast_ss(a_stream + 2);

        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c12 = _mm256_fmadd_ps(a1, b2, c12);
        c13 = _mm256_fmadd_ps(a1, b3, c13);

        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c22 = _mm256_fmadd_ps(a2, b2, c22);
        c23 = _mm256_fmadd_ps(a2, b3, c23);

        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c32 = _mm256_fmadd_ps(a3, b2, c32);
        c33 = _mm256_fmadd_ps(a3, b3, c33);

        a_stream += 3;
        b_stream += 24;
    }

    _mm256_storeu_ps(c.data.data()+(ii*c.columns)+jj,c11);
    _mm256_storeu_ps(c.data.data()+((ii+1)*c.columns)+jj,c21);
    _mm256_storeu_ps(c.data.data()+((ii+2)*c.columns)+jj,c31);

    _mm256_storeu_ps(c.data.data()+(ii*c.columns)+(jj+8),c12);
    _mm256_storeu_ps(c.data.data()+((ii+1)*c.columns)+(jj+8),c22);
    _mm256_storeu_ps(c.data.data()+((ii+2)*c.columns)+(jj+8),c32);

    _mm256_storeu_ps(c.data.data()+(ii*c.columns)+(jj+16), c13);
    _mm256_storeu_ps(c.data.data()+((ii+1)*c.columns)+(jj+16),c23);
    _mm256_storeu_ps(c.data.data()+((ii+2)*c.columns)+(jj+16), c33);
}

inline void micro_kernel3x24_safe(float* a_stream, float* b_stream, Matrix& c, int K, int ii, int jj) {
    alignas(32) float temp[72];

    __m256 c11 = _mm256_setzero_ps();
    __m256 c12 = _mm256_setzero_ps();
    __m256 c13 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c22 = _mm256_setzero_ps();
    __m256 c23 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c32 = _mm256_setzero_ps();
    __m256 c33 = _mm256_setzero_ps();

    #pragma gcc unroll 4
    for (int k = 0; k < K; k++) {
        __m256 b1 = _mm256_loadu_ps(b_stream);
        __m256 b2 = _mm256_loadu_ps(b_stream + 8);
        __m256 b3 = _mm256_loadu_ps(b_stream + 16);

        __m256 a1 = _mm256_broadcast_ss(a_stream);
        __m256 a2 = _mm256_broadcast_ss(a_stream + 1);
        __m256 a3 = _mm256_broadcast_ss(a_stream + 2);

        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c12 = _mm256_fmadd_ps(a1, b2, c12);
        c13 = _mm256_fmadd_ps(a1, b3, c13);

        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c22 = _mm256_fmadd_ps(a2, b2, c22);
        c23 = _mm256_fmadd_ps(a2, b3, c23);

        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c32 = _mm256_fmadd_ps(a3, b2, c32);
        c33 = _mm256_fmadd_ps(a3, b3, c33);

        a_stream += 3;
        b_stream += 24;
    }

    _mm256_store_ps(temp, c11);
    _mm256_store_ps(temp + 8, c12);
    _mm256_store_ps(temp + 16, c13);
    _mm256_store_ps(temp + 24, c21);
    _mm256_store_ps(temp + 32, c22);
    _mm256_store_ps(temp + 40, c23);
    _mm256_store_ps(temp + 48, c31);
    _mm256_store_ps(temp + 56, c32);
    _mm256_store_ps(temp + 64, c33);

    int valid_rows = std::min(3, c.rows - ii);
    int valid_cols = std::min(24, c.columns - jj);

    for (int i = 0; i < valid_rows; i++) {
        for (int j = 0; j < valid_cols; j++) {
            c.data[(ii + i) * c.columns + (jj + j)] = temp[i * 24 + j];        
        }
    }
}

Matrix multiply(Matrix a, Matrix b) {
    Matrix c(a.rows, b.columns);

    std::vector<float> a_packed(pad_height(a.rows, 3)*a.columns, 0.0f);
    int aheight = pad_height(a.rows, 3);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.columns; j++) {
            int index = (i/3)*3*a.columns + 3*j + (i % 3);
            a_packed[index] = a.data[i*a.columns+j];
        }
    }

    int bwidth = pad_width(b.columns, 24);
    std::vector<float> b_packed(b.rows*bwidth, 0.0f);
    for (int i = 0; i < b.rows; i++) {
        for (int j = 0; j < b.columns; j++) {
            int index = (j / 24) * b.rows * 24 + (24 * i) + (j % 24);
            b_packed[index] = b.data[i*b.columns+j];
        }
    }

    int safe_rows = c.rows - (c.rows % 3);
    int safe_columns = c.columns - (c.columns % 24);

    for (int jj = 0; jj < safe_columns; jj += 24) {
        for (int ii = 0; ii < safe_rows; ii += 3) {
            micro_kernel3x24_unsafe(&a_packed[ii], b_packed.data()+jj, c, a.columns, ii, jj);
        }
    }

    for (int jj = safe_columns; jj < c.columns; jj += 24) {
        for (int ii = 0; ii < safe_rows; ii += 3) {
            micro_kernel3x24_safe(&a_packed[ii], b_packed.data()+jj, c, a.columns, ii, jj);
        }
    }

    for (int jj = 0; jj < c.columns; jj += 24) {
        for (int ii = safe_rows; ii < c.rows; ii += 3) {
            micro_kernel3x24_safe(&a_packed[ii], b_packed.data()+jj, c, a.columns, ii, jj);
        }
    }

    return c;
}