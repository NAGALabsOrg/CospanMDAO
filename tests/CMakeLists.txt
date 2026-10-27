# The cospanmdao test package, standalone like SlabKit's tests: it finds the
# INSTALLED library (find_package(cospanmdao)) and is not part of the root
# project (the root CMakeLists does not add_subdirectory(tests)). Built by
# pixi-build as the package `cospanmdao-tests`; run with `pixi run test`.
#
# tests/spike is a separate, standalone project and is not referenced from here.
# (tests/smoke, the pre-section-11 smoke programs, was deleted in P8: every one
# of them used removed API and the suite covers the same ground.)
cmake_minimum_required(VERSION 3.24)
project(cospanmdao-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AdaptiveCpp's exported acpp-rt links Threads::Threads and does not find it
# itself, so Threads comes first.
find_package(Threads REQUIRED)
find_package(AdaptiveCpp REQUIRED)
find_package(cospanmdao REQUIRED)
find_package(Catch2 3 REQUIRED)

set(COSPAN_TEST_SOURCES
    # test_chunked.cpp, test_leading_range.cpp, test_replay.cpp: chunked systems, kept for the redesign (not built)
    test_component.cpp
    test_derivatives.cpp
    test_device_scan.cpp
    test_events.cpp
    test_flavors.cpp
    test_fused_groups.cpp
    test_host_access.cpp
    test_init_shapes.cpp
    test_model.cpp
    test_p4.cpp
    test_r04.cpp
    test_r05.cpp
    test_ragged.cpp
    test_ragged_ties.cpp
    test_range_split.cpp
    test_rework.cpp
    test_shapes.cpp
    test_solvers.cpp
    test_diagnostics.cpp
    test_epochs.cpp
    test_system.cpp
    # The dynamic-function probe: the kernels in one translation unit, the
    # SYCL_EXTERNAL definitions in the other.
    test_acpp_dynfn.cpp
    acpp_dynfn_defs.cpp
)

add_executable(cospan-tests ${COSPAN_TEST_SOURCES})
target_link_libraries(cospan-tests PRIVATE Catch2::Catch2WithMain naga::cospan)
# Header-only library: the kernels are emitted by these translation units.
add_sycl_to_target(TARGET cospan-tests SOURCES ${COSPAN_TEST_SOURCES})

install(TARGETS cospan-tests RUNTIME DESTINATION bin)
