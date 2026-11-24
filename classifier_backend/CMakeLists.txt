project(rena-classifier-backend)
set(RENA_CLASSIFIER_BACKEND_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include"
    CACHE STRING "rena classifier_backend include directory"
    FORCE
)
set(RENA_CLASSIFIER_BACKEND_SRC_LIST
    ${PROJECT_SOURCE_DIR}/src/builtin/embedding_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/semantic/embedding_gemma.cpp
    ${PROJECT_SOURCE_DIR}/src/semantic/llama_backend.cpp
    ${PROJECT_SOURCE_DIR}/src/semantic/multilingual_e5.cpp
)
add_library(${PROJECT_NAME} SHARED
    ${RENA_CLASSIFIER_BACKEND_SRC_LIST}
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${RENA_CORE_INCLUDE_DIR}
        ${RENA_CLASSIFIER_BACKEND_INCLUDE_DIR}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        rena-core
    PRIVATE
        llama
)
