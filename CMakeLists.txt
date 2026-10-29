cmake_minimum_required(VERSION 3.16)

project(fast_rag LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(fast_rag
    main.cpp
    doc_utilities.cpp
)
target_compile_features(fast_rag PRIVATE cxx_std_20)
set_target_properties(fast_rag PROPERTIES
    CXX_EXTENSIONS OFF
)
target_link_libraries(fast_rag PRIVATE MPI::MPI_CXX)
