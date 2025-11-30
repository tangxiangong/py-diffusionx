cpp-build:
    cd cpp/diffusionx && vcpkg install
    cd cpp && rm -rf build && mkdir build && cd build && cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=/Users/xiaoyu/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release .. && cmake --build .
