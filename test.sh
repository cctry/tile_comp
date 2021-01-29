nvcc -O3 -arch=sm_70 tile_size/tile_size_test.cu tile_size/cublas.cu lib/utils.cu -Ilib -lcublas -DTIME
./a.out