// Copyright (C) 2024 Advanced Micro Devices, Inc
//
// SPDX-License-Identifier: MIT

#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include "ap_int.h"

//Array size to access
#define DATA_SIZE 256

typedef ap_int<8> buf_t;

//Declaring the hardware function
void matrix_mult(int* in1 , int* in2 , int* out);

#endif
