// Copyright (C) 2024 Advanced Micro Devices, Inc
//
// SPDX-License-Identifier: MIT

#include "matrix_mult_fabric.hpp"

void matrix_mult(int* in1, int* in2, int* out)
{
    buf_t Arow[DATA_SIZE];
    buf_t Bbuf[DATA_SIZE][DATA_SIZE];

#pragma HLS INTERFACE m_axi depth=DATA_SIZE*DATA_SIZE port=in1
#pragma HLS INTERFACE m_axi depth=DATA_SIZE*DATA_SIZE port=in2
#pragma HLS INTERFACE m_axi depth=DATA_SIZE*DATA_SIZE port=out
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

#pragma HLS bind_storage variable=Bbuf type=ram_2p

#pragma HLS array_partition variable=Arow complete dim=1
#pragma HLS array_partition variable=Bbuf cyclic factor=128 dim=1

    READ_B:
    for (int idx = 0; idx < DATA_SIZE * DATA_SIZE; idx++){
        #pragma HLS PIPELINE II=1
        int i = idx / DATA_SIZE;
        int j = idx % DATA_SIZE;
        Bbuf[i][j] = in2[idx];
    }

    for (int i = 0; i < DATA_SIZE; i++){
        LOAD_A_ROW:
        for (int k = 0; k < DATA_SIZE; k++){
            #pragma HLS PIPELINE
            Arow[k] = in1[i * DATA_SIZE + k];
        }

        COMPUTE:
        for (int j = 0; j < DATA_SIZE; j++){
            int tmp = 0;
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < DATA_SIZE; k++){
                #pragma HLS UNROLL
                tmp += Arow[k] * Bbuf[k][j];
                #pragma HLS bind_op variable=tmp op=add impl=fabric
            }
            out[i * DATA_SIZE + j] = tmp;
        }
    }
}
