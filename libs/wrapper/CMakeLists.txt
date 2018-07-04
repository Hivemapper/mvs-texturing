file (GLOB HEADERS "*.h")
file (GLOB SOURCES "*.cpp")

set(LIBRARY mvs_tex_wrapper)
add_library(${LIBRARY} STATIC SHARED ${SOURCES})
#add_dependencies(${LIBRARY} mvs_tex)
target_link_libraries(${LIBRARY} mvs_tex -lmve -lmve_util)

install(TARGETS ${LIBRARY} LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/mvs_tex/wrapper)
