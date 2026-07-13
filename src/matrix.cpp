#include "matrix.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>

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
        std::fprintf(stderr, "parse_next_float: unexpected EOF\n");
        std::abort();
    }

    while (p < end && (*p >= '0' && *p <= '9')) {
        number = number*10 + (*p - '0');
        p++;
    }

    ptr = p;
    return number;
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

