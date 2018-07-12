file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

set(LIBRARY mvs_tex)
add_library(${LIBRARY} SHARED ${SOURCES})
set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
add_dependencies(${LIBRARY} ext_mve ext_rayint ext_mapmap)
#add_dependencies(${LIBRARY} ext_mve ext_rayint ext_eigen ext_mapmap)
target_link_libraries(${LIBRARY} ${TBB_LIBRARIES} mvs_tex_mask -lmve -lmve_util ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES})

install(TARGETS ${LIBRARY} LIBRARY DESTINATION lib)
