#!/bin/bash

mkdir -p build
cd build
cmake ../
make -j2
mv main ..
