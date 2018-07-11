file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

include_directories(/usr/local/include/eigen3)

set(BIN texrecon)

add_executable(${BIN} ${SOURCES})
set_property(TARGET ${BIN} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
#add_dependencies(${BIN} ext_mve)
target_link_libraries(${BIN} mvs_tex_wrapper -lmve -lmve_util)
#target_link_libraries(${BIN} mvs_tex ${TBB_LIBRARIES} -lmve -lmve_util)
install(TARGETS ${BIN} RUNTIME DESTINATION bin)
