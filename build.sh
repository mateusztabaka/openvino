set -e
export DLDT=$HOME/dldt_dist_2021.2
[[ ! -e build ]] && mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX=$DLDT -DENABLE_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCV=OFF
make -j `nproc` install
popd
