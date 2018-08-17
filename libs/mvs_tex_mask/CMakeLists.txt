file (GLOB HEADERS "*.h")
file (GLOB SOURCES "*.cpp")

set(VERSION 1.0)
set(SOVERSION 1.0)

set(LIBRARY mvs_tex_mask)

add_library(${LIBRARY} SHARED ${SOURCES})
SET_TARGET_PROPERTIES(${LIBRARY} PROPERTIES SOVERSION ${MVS_TEX_VERSION})
#add_dependencies(${LIBRARY} ext_eigen)
#target_link_libraries(${LIBRARY} mvs_texturing)

install(TARGETS ${LIBRARY} LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/mvs_tex_mask)
