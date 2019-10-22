#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <util/timer.h>
#include <util/system.h>
#include <util/file_system.h>
#include <mve/mesh_io_ply.h>


#include "mvs_tex_wrapper/wrapper.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << std::string {argv[0]} << " PATH/TO/PLY" << std::endl;
    return 1;
  }
  
  std::string path {argv[1]};
  
  mve::TriangleMesh::Ptr mesh {};

  try {
    mesh = mve::geom::load_ply_mesh(path);
    
    std::cout << "vertex count: " << mesh->get_vertices().size() << std::endl;
    std::cout << "face count: " << mesh->get_faces().size() << std::endl;

    if (mesh->has_vertex_normals()) {
      std::cout << "vertex normals: " << mesh->get_vertex_normals().size() << std::endl;
    }
  }
  
  catch (std::exception& e) {
    std::cerr << "Oops: " << e.what() << std::endl;
  }

  catch (...) {
    std::cerr << "Oops: unknown cause" << std::endl;
  }

  return 0;
}
