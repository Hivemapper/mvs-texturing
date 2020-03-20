externalproject_add(ext_mapmap
    PREFIX          ext_mapmap
    GIT_REPOSITORY  https://github.com/dthuerck/mapmap_cpu.git
    GIT_TAG         6ea08714e1865fe73afff8892c7136cd1fe1e6f3
    UPDATE_COMMAND  ""
    SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/mapmap
    CONFIGURE_COMMAND ""
    BUILD_COMMAND   ""
    INSTALL_COMMAND ""
)

externalproject_add(ext_rayint
    PREFIX          ext_rayint
    GIT_REPOSITORY  https://github.com/nmoehrle/rayint.git
    GIT_TAG         d5a8126aebb98c68b2b98bea05fcc8cebb6a2fb9
    UPDATE_COMMAND  ""
    SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/rayint
    CONFIGURE_COMMAND ""
    BUILD_COMMAND   ""
    INSTALL_COMMAND ""
)

#externalproject_add(ext_eigen
#    PREFIX          ext_eigen
#    URL             https://bitbucket.org/eigen/eigen/get/3.3.2.tar.gz
#    URL_MD5         36b5731ab7d7e0c10843ac93bd9fd270
#    SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/eigen
#    CONFIGURE_COMMAND ""
#    BUILD_COMMAND   ""
#    INSTALL_COMMAND ""
#)

externalproject_add(ext_mve
    PREFIX          mve
    GIT_REPOSITORY  https://github.com/Hivemapper/mve.git
    GIT_TAG         3f62a9b8a26488b9a6eb27e91f578e3a8350f6af
    UPDATE_COMMAND  ""
    SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/mve
    CONFIGURE_COMMAND ""
    BUILD_COMMAND   make -C libs/mve && make -C libs/util #not platform independent
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)
