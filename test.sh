rm -rf ./build
mkdir build
cd build
cmake --config Release .. 
make -j
./tile_comp > ../res.txt 
if (( $EUID != 0 ));
then
  sudo $(which nvprof) ./tile_comp_prof > ../prof.txt
fi