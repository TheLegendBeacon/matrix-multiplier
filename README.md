# Matrix Multiplication.

## Aim:
### Given two matrices A and B in row-major format, output AxB. Leave no stone unturned in the pursuit of performance.

## Method of Input:
- The matrices are input in row-major format (row followed by column, followed by the matrix itself.) For example,
```
2 3
1 1 1
1 1 1

3 4
1 1 1 1
1 1 1 1
1 1 1 1
```
- The input file is `mmap`ped to memory, on which a `float` parser runs. 
- The elements are stored in the `Matrix` struct.

## Hardware Notes:
- This project was made with an Intel 13th Gen CPU in mind. 
- Specifically, since Raptor Lake CPUs do not have support for AVX-512 (`:(`) this is limited to AVX-2 instructions at most.

## Method of Multiplication
The idea is to use the regular O(n^3) naive algorithm, and improve as we go.

## References
- [Achieving Peak Performance for Matrix Multiplication in C++](https://www.youtube.com/watch?v=CeoGWwaL8CY)