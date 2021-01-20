cmake_minimum_required(VERSION 2.8)
project( VSA )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH  		"${PROJECT_SOURCE_DIR}/cmake")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arch64")
  set(CMAKE_CXX_FLAGS 		"-std=c++14 -DIL_STD -pthread") # "-Wall -Wextra" for strict warnings check
else()
  set(CMAKE_CXX_FLAGS 		"-std=c++14 -DIL_STD -m64 -pthread") # "-Wall -Wextra" for strict warnings check
endif()

set(CMAKE_CXX_FLAGS_DEBUG 	"-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -flto")
set(LINKER_OPTIONS 			"-flto -Wl,--no-as-needed")
set(DARKNET_PATH 			"${PROJECT_SOURCE_DIR}/darknet")
set(DARKNET_LIB_NAME 		"libdarknet")

# compile definitions
add_compile_definitions(OPENCV)
# Find packges
find_package( OpenCV 4 REQUIRED )

if (CONTROLLER_TYPE MATCHES "mpc")
  add_compile_definitions(MPC_CONTROLLER)
  if (MPC_SOLVER MATCHES "cplex")
    find_package(CPLEX) # Define CPLEX_STUDIO_DIR to cplex_studio* path
    set(CPLEX_LIBS 				cplex-library cplex-concert ilocplex cplex-cp dl)
    add_compile_definitions(CPLEX_SOLVER)
  else()
    find_package(GLPK)
  endif()
elseif(CONTROLLER_TYPE MATCHES "basic")
  add_compile_definitions(BASIC_CONTROLLER)
elseif(CONTROLLER_TYPE MATCHES "pid")
  add_compile_definitions(PID_CONTROLLER)
endif()

set(PROJECT_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Include files
include_directories(
  ${PROJECT_SRC_PATH}
  ${PROJECT_SRC_PATH}/json
  ${PROJECT_SRC_PATH}/controller
	${OpenCV_INCLUDE_DIRS} 
	${DARKNET_PATH}/include
	${CPLEX_INCLUDE_DIR}
	${CPLEX_CONCERT_INCLUDE_DIR}
	${CPLEX_CP_INCLUDE_DIR}
	${GLPK_INCLUDE_DIRS}
)

add_library(${DARKNET_LIB_NAME} SHARED IMPORTED)


#set_property(TARGET ${DARKNET_LIB_NAME} PROPERTY IMPORTED_LOCATION "${DARKNET_PATH}/libdarknet.so")
set_target_properties(${DARKNET_LIB_NAME} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${DARKNET_PATH}/src"
        IMPORTED_LOCATION "${DARKNET_PATH}/libdarknet.so"
)
message(${DARKNET_PATH})

if (CLIENT)
  # compile executables
  add_executable( client ${PROJECT_SOURCE_DIR}/src/client.cpp )

  # link executables
  target_link_libraries( client ${OpenCV_LIBS} ${CPLEX_LIBS} ${GLPK_LIBRARIES})
endif()

if (SERVER)
  # compile executables
  add_executable( server ${PROJECT_SOURCE_DIR}/src/server.cpp )

  # link executables
  target_link_libraries( server ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
endif()


if (TEST)
  # Test executables
  if (CONTROLLER_TYPE MATCHES "mpc")
    if (MPC_SOLVER MATCHES "cplex")
      add_executable(test_cplex ${PROJECT_SOURCE_DIR}/src/test_cplex.cpp)
      target_link_libraries(test_cplex PRIVATE ${CPLEX_LIBS})
    else()
      add_executable(test_glpk ${PROJECT_SOURCE_DIR}/src/test_glpk.cpp)
      target_link_libraries(test_glpk PRIVATE ${GLPK_LIBRARIES})
    endif()
  endif()

  add_executable( detection ${PROJECT_SOURCE_DIR}/src/tests/detection.cpp )
  add_executable( detection_timing ${PROJECT_SOURCE_DIR}/src/tests/detection_timing.cpp )
  target_link_libraries( detection ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
  target_link_libraries( detection_timing ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )

endif()
