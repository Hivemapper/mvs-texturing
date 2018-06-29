file (GLOB HEADERS "*.h")
file (GLOB SOURCES "*.cpp")

set(LIBRARY texture_wrapper)
add_library(${LIBRARY} SHARED ${SOURCES})
#add_dependencies(${LIBRARY} mvs_texturing)
target_link_libraries(${LIBRARY} mvs_texturing)

install(TARGETS ${LIBRARY} LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/texture_wrapper)
