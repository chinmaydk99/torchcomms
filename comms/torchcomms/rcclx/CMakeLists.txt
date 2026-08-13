# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_rcclx
file(GLOB TORCHCOMMS_RCCLX_SOURCES "comms/torchcomms/rcclx/*.cpp")

# RcclxApiShardedRelay.cpp and RcclxApiShardedRelayStub.cpp both define the same
# DefaultRcclxApi::* methods; the glob above pulls in both, which duplicate-links.
# (BUCK selects one via select(); CMake has no equivalent.) Compile exactly one:
# the stub (default) links against a stock librccl, while the real relay TU needs
# the extended RCCL fork that provides ncclShardedRelay*/ncclx::* symbols.
option(USE_RCCLX_SHARDED_RELAY
    "Link the Meta extended RCCL sharded-relay TU (requires the extended fork librccl)" OFF)
foreach(_src ${TORCHCOMMS_RCCLX_SOURCES})
    get_filename_component(_fname ${_src} NAME)
    if(NOT USE_RCCLX_SHARDED_RELAY AND _fname STREQUAL "RcclxApiShardedRelay.cpp")
        list(REMOVE_ITEM TORCHCOMMS_RCCLX_SOURCES ${_src})
    elseif(USE_RCCLX_SHARDED_RELAY AND _fname STREQUAL "RcclxApiShardedRelayStub.cpp")
        list(REMOVE_ITEM TORCHCOMMS_RCCLX_SOURCES ${_src})
    endif()
endforeach()

set(CONDA_INCLUDE "${CONDA_PREFIX}/include")
# User can override via ROCM_HOME environment variable; otherwise use
# ROCM_PATH set by LoadHIP.cmake, included via find_package(Torch).
if(DEFINED ENV{ROCM_HOME})
    set(ROCM_HOME $ENV{ROCM_HOME})
elseif(ROCM_PATH)
    set(ROCM_HOME ${ROCM_PATH})
else()
    message(FATAL_ERROR "ROCM_PATH is not set. Ensure PyTorch is installed and find_package(Torch) is called before this point.")
endif()
set(ROCM_INCLUDE "${ROCM_HOME}/include")
set(RCCLX_INCLUDE $ENV{RCCLX_INCLUDE})
if(NOT RCCLX_INCLUDE)
    set(RCCLX_INCLUDE "${ROCM_HOME}/include/rccl")
endif()
message(STATUS "RCCLX_INCLUDE is set to ${RCCLX_INCLUDE}")
if(DEFINED ENV{RCCLX_LIB})
    set(RCCLX_SHARED_LIB $ENV{RCCLX_LIB}/librccl.so)
else()
    set(RCCLX_SHARED_LIB "${ROCM_HOME}/lib/librccl.so")
endif()
message(STATUS "RCCLX_SHARED_LIB is set to ${RCCLX_SHARED_LIB}")

add_library(torchcomms_comms_rcclx MODULE ${TORCHCOMMS_RCCLX_SOURCES})
set_target_properties(torchcomms_comms_rcclx PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_rcclx"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)

target_include_directories(torchcomms_comms_rcclx PRIVATE
    ${ROOT}
    ${RCCLX_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${ROCM_INCLUDE}
    ${ROCM_INCLUDE}/hip
)
torchcomms_use_torch_pybind11(torchcomms_comms_rcclx)
target_compile_features(torchcomms_comms_rcclx PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_rcclx PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_rcclx PRIVATE
    torchcomms
    glog::glog
    fmt::fmt
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    ${RCCLX_SHARED_LIB}
)

set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} -std=c++20")

install(TARGETS torchcomms_comms_rcclx
        LIBRARY DESTINATION .
)
