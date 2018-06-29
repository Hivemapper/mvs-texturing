file (GLOB HEADERS "*.h")
file (GLOB SOURCES "*.cpp")

set(LIBRARY mvs_tex_mask)
add_library(${LIBRARY} SHARED ${SOURCES})
add_dependencies(${LIBRARY} ext_eigen)
target_link_libraries(${LIBRARY} mvs_texturing)

install(TARGETS ${LIBRARY} LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/mvs_tex/mask)
