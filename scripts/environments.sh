#!/bin/bash
# Giving some args to accelerate compilation
export BUILD_TEST=0
export USE_CUDA=0
# Modify these to suit your system
export CC=gcc-11
export CXX=g++-11
export MAX_JOBS=20
export SYSROOT=/root/sysroot
export CFLAGS="--sysroot=$SYSROOT"
export CXXFLAGS="--sysroot=$SYSROOT"
export LDFLAGS="--sysroot=$SYSROOT"
export CMAKE_FIND_ROOT_PATH=$SYSROOT
export CMAKE_SYSROOT=$SYSROOT
