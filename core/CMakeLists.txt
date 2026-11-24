project(rena-core)

set(RENA_CORE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include"
    CACHE STRING "rena core include directory"
    FORCE
)
set(RENA_CORE_SRC_LIST
    ${PROJECT_SOURCE_DIR}/src/log.cpp
)
add_library(${PROJECT_NAME} SHARED
    ${RENA_CORE_SRC_LIST}
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${RENA_CORE_INCLUDE_DIR}
)
