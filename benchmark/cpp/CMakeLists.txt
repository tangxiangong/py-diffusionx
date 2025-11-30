cmake_minimum_required(VERSION 3.31)
project(cpp-bench)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# macOS 特定配置
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MACOS_SDK_PATH}")
endif()

# 将 diffusionx 的 vcpkg_installed 目录添加到 CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/diffusionx/vcpkg_installed/arm64-osx")

add_subdirectory(diffusionx)

# 添加 include 目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 为 src/ 下的每个 .cpp 文件生成一个可执行程序
file(GLOB_RECURSE ALL_SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

foreach(src_file ${ALL_SRC_FILES})
    get_filename_component(executable_name ${src_file} NAME_WE)
    add_executable(${executable_name} ${src_file})
    target_link_libraries(${executable_name} PRIVATE diffusionx)
endforeach()

add_custom_command(
    OUTPUT "${PROJECT_SOURCE_DIR}/compile_commands.json"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${PROJECT_SOURCE_DIR}/compile_commands.json"
    COMMENT "Copying compile_commands.json to project root"
)
