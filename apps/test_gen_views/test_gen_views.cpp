//
// Test Harness Created by David Hattery on 4/19/19.
//

#include "tex/texturing.h"
#include "mvs_tex_wrapper/wrapper.h"
#include "tex/texture_view.h"
#include <mve/mesh_io_ply.h>
#include <util/timer.h>

// This is a copy of same named method inside wrapper.cpp since it isn't public
bool is_valid_tri(std::size_t i, const std::vector<bool>& mask, const std::vector<unsigned int>& old_faces) {
  return mask[old_faces[i*3]] && mask[old_faces[i*3+1]] && mask[old_faces[i*3+2]];
}

// This is a copy of same named method inside wrapper.cpp since it isn't public
void generate_face_reindex(const std::vector<bool>& mask,
                           const std::vector<unsigned int>& old_faces,
                           std::vector<std::size_t>& new_indices) {
  new_indices.resize(old_faces.size()/3);
  std::size_t front = 0;
  std::size_t back = new_indices.size() -1;
  while (front < back) {
    if (is_valid_tri(front, mask, old_faces)) {
      new_indices[front] = front;
      ++front;
    } else {
      while (front < back && !is_valid_tri(back, mask, old_faces)) {
        new_indices[back] = std::numeric_limits<std::size_t>::max();
        --back;
      }
      if (back > front && is_valid_tri(back, mask, old_faces)) {
        new_indices[back] = front;
        back--;
        front++;
      }
    }
  }
}


// This method mimics external call to wrapper.cpp from HM
int main(){
  util::WallTimer timer {};
  tex::TextureViews texture_views {};
  int texture_channels = 0;
  MvsTexturing::TextureSettings texture_settings {};

  texture_settings.do_geometric_visibility_test = true;
  texture_settings.do_keep_unseen_faces = false;
  texture_settings.do_gamma_tone_mapping = false;
  int num_colors = 3;

//  if (single_model_mode
//      || (use_vm_masking && !vm_masking_succeeded)) {
  texture_settings.do_geometric_visibility_test = true;
  texture_settings.do_keep_unseen_faces = false;
//  }

  std::string data_cost_file;
  std::string labeling_file;

  const std::shared_ptr<MvsTexturing::EuclideanViewMask> ev_mask;
// //  const MvsTexturing::EuclideanViewMask ev_mask;
//  uint atlas_size = 0;

  const std::string scene_directory="/home/vagrant/redo_classes/5b56c067ad391464ac5779b0/scene";
  const std::string input_mesh="/home/vagrant/redo_classes/5b56c067ad391464ac5779b0/mesh.ply";
  const std::string out_prefix="/home/vagrant/redo_classes/5b56c067ad391464ac5779b0/5b56c067ad391464ac5779b0";


  std::vector<std::vector<bool>> sub_vert_masks {};

//  // determine splits if needed
//  std::vector<std::vector<bool>> sub_vertex_masks {};
// //  const std::vector<std::string> sub_names={"one"};
  std::vector<std::string> sub_names {};
  std::string output_mesh {};
//  int split_level = 1;
//  int normalization = -1;
//  createSplits(input_mesh, split_level, normalization, vertex_masks, sub_names);

//  ComputeCloud::Cloud tmp_mesh(input_mesh);
  sub_names.emplace_back("");
  // NOTE this integer is specific to test mesh only
  sub_vert_masks.emplace_back(155285, true);

//  ComputeCloud::Cloud tmp_mesh {mesh_file_i};
  std::vector<std::vector<uint8_t>> segmentation_classes {};
//  segmentation_classes.emplace_back(tmp_mesh.getSize(), 0); // set all to unknown class
  segmentation_classes.emplace_back(155285, 0);

//  textureMesh(texture_settings, scene_directory, input_mesh, out_prefix,
//      sub_vert_masks, sub_names, ev_mask, atlas_size);

  tex::generate_texture_views(scene_directory, &texture_views, "tmp");
  if (!texture_views.empty()) {
    texture_channels = texture_views[0].get_channels();
  }
  mve::TriangleMesh::Ptr mesh {};
  try {
    mesh = mve::geom::load_ply_mesh(input_mesh);
  } catch (std::exception& e) {
    std::cerr << "\tCould not load mesh: " << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  mve::MeshInfo mesh_info(mesh);
  tex::prepare_mesh(&mesh_info, mesh);

  std::size_t const num_faces = mesh->get_faces().size() / 3;

  std::cout << "Building adjacency graph: " << std::endl;
  tex::Graph graph(num_faces);
  tex::build_adjacency_graph(mesh, mesh_info, &graph);

  //
  // Build Processing Settings
  //
  tex::Settings settings {};
//  settings.local_seam_leveling = false;
  if (texture_settings.do_use_gmi_term) {
    settings.data_term = tex::DATA_TERM_GMI;
  } else {
    settings.data_term = tex::DATA_TERM_AREA;
  }
  if (texture_settings.do_gauss_clamping) {
    settings.outlier_removal = tex::OUTLIER_REMOVAL_GAUSS_CLAMPING;
  } else if (texture_settings.do_gauss_damping) {
    settings.outlier_removal = tex::OUTLIER_REMOVAL_GAUSS_DAMPING;
  } else {
    settings.outlier_removal = tex::OUTLIER_REMOVAL_NONE;
  }
  if (texture_settings.do_gamma_tone_mapping) {
    settings.tone_mapping = tex::TONE_MAPPING_GAMMA;
  } else {
    settings.tone_mapping = tex::TONE_MAPPING_NONE;
  }
  settings.geometric_visibility_test = texture_settings.do_geometric_visibility_test;
  settings.global_seam_leveling = texture_settings.do_global_seam_leveling;
  settings.local_seam_leveling = texture_settings.do_local_seam_leveling;
  settings.local_seam_leveling = true;
  settings.hole_filling = texture_settings.do_hole_filling;
  settings.keep_unseen_faces = texture_settings.do_keep_unseen_faces;


  if (labeling_file.empty()) {
    std::cout << "View selection:" << std::endl;

    tex::DataCosts data_costs(num_faces, texture_views.size());
    if (data_cost_file.empty()) {
      // std::cout << "- added - Calculating Data costs" << std::endl;
      tex::calculate_data_costs(mesh, &texture_views, settings, &data_costs, ev_mask);

    } else {
      std::cout << "\tLoading data cost file... " << std::flush;
      try {
        tex::DataCosts::load_from_file(data_cost_file, &data_costs);
      } catch (util::FileException &e) {
        std::cout << "failed!" << std::endl;
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::cout << "done." << std::endl;
    }

    std::cout << "- added - Selecting Views" << std::endl;
    try {
      tex::view_selection(data_costs, &graph, settings);
    } catch (std::runtime_error& e) {
      std::cerr << "\tOptimization failed: " << e.what() << std::endl;
      std::exit(EXIT_FAILURE);
    }

  } else {
    std::cout << "Loading labeling from file... " << std::flush;
  }

  // generate full texture patches
  tex::TexturePatches texture_patches {};
  tex::TexturePatches texture_object_class_patches {};
  {
    util::WallTimer rwtimer {};
    /* Create texture patches and adjust them. */
    tex::VertexProjectionInfos vertex_projection_infos {};
    std::cout << "Generating texture patches:" << std::endl;
    tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
                                  settings, &vertex_projection_infos, &texture_patches);

    if (settings.global_seam_leveling) {
      std::cout << "Running global seam leveling:" << std::endl;
      tex::global_seam_leveling(graph, mesh, mesh_info, vertex_projection_infos, &texture_patches);
    } else {
#pragma omp parallel for schedule(dynamic)
      for (std::size_t i = 0; i < texture_patches.size(); ++i) {
        TexturePatch::Ptr texture_patch = texture_patches[i];
        std::vector<math::Vec3f> patch_adjust_values(texture_patch->get_faces().size() * 3, math::Vec3f(0.0f));
        texture_patch->adjust_colors(patch_adjust_values);
      }
    }

    if (texture_channels > num_colors) {
      std::cout << "Making object class textures:" << std::endl;
      // Build a copy of texture_patches for object classes
      for (const auto &texture_patch : texture_patches) {
        texture_object_class_patches.emplace_back(texture_patch->duplicate());
      }

      std::cout << "Building object class texture image:" << std::endl;
#pragma omp parallel for schedule(dynamic)
      for (std::size_t i = 0; i < texture_object_class_patches.size(); ++i) {
        TexturePatch::Ptr texture_object_class_patch = texture_object_class_patches[i];
        std::vector<math::Vec3f>
            patch_adjust_values(texture_object_class_patch->get_faces().size() * 3, math::Vec3f(0.0f));
        texture_object_class_patch->adjust_colors(patch_adjust_values, texture_channels);
      }
      std::cout << "\tTook: " << rwtimer.get_elapsed_sec() << "s" << std::endl;

      // For n-channel, get texture_patches for object classes here too--map classes to rgb
      if (settings.local_seam_leveling) {
        util::WallTimer seamtimer {};
        std::cout << "Running local seam leveling with object classes:" << std::endl;
        tex::local_seam_leveling_n(graph, mesh, vertex_projection_infos, &texture_patches, texture_channels, &texture_object_class_patches);
//        std::cout << "Running local seam leveling ignoring object classes:" << std::endl;
//        tex::local_seam_leveling_n(graph, mesh, vertex_projection_infos, &texture_patches, texture_channels);
        std::cout << "\tSeam leveling with object classes took: " << seamtimer.get_elapsed_sec() << "s" << std::endl;
      }

      /* Sample vertex colors. */
      std::cout << "Setting segmentation class probabilities:" << std::endl;
      segmentation_classes.clear();
      for (std::size_t i = 0; i < vertex_projection_infos.size(); ++i) {
        std::vector<tex::VertexProjectionInfo> const & projection_infos = vertex_projection_infos[i];
        for (tex::VertexProjectionInfo const &projection_info : projection_infos) {
          TexturePatch::Ptr texture_patch = texture_patches.at(projection_info.texture_patch_id);
          if (texture_patch->get_label() == 0) {
            continue;
          }
          auto pixel_channels = texture_patch->get_pixel_value_n(projection_info.projection, texture_channels);
          for (auto&& pixel : pixel_channels) {
            pixel *= 255.0f;
          }
          segmentation_classes.emplace_back(pixel_channels.begin() + num_colors, pixel_channels.end());
        }
      }

    } else {
      if (settings.local_seam_leveling) {
        util::WallTimer seamtimer {};
        std::cout << "Running local seam leveling:" << std::endl;
        tex::local_seam_leveling(graph, mesh, vertex_projection_infos, &texture_patches);
        std::cout << "\tSeam leveling took: " << seamtimer.get_elapsed_sec() << "s" << std::endl;
      }
    }

  }

  // Now loop, generating+saving subindexed meshes and atlas
#pragma omp parallel for schedule(dynamic)
  for (std::size_t vi = 0; vi < sub_vert_masks.size(); ++vi) {
    util::WallTimer modeltimer {};
    std::cout << "\nFinalizing Sub-Model " << sub_names[vi] << " - " << vi+1 << " of " << sub_vert_masks.size() << std::endl;
    tex::TextureAtlases sub_texture_atlases {};
    tex::TextureAtlases sub_texture_object_class_atlases {};
    const std::vector<bool>& vertex_mask {sub_vert_masks[vi]};
    std::vector<bool> inverted_mask(vertex_mask.size());
    for (std::size_t i = 0; i < vertex_mask.size(); ++i) {
      inverted_mask[i] = !vertex_mask[i];
    }

    const std::string& sub_name {sub_names[vi]};
    std::vector<std::size_t> face_indices {};
    // generate face reindex
    generate_face_reindex(vertex_mask, mesh->get_faces(), face_indices);
    // redo mesh
    mve::TriangleMesh::Ptr sub_mesh = mesh->duplicate();
    sub_mesh->delete_vertices_fix_faces(inverted_mask);

    if (sub_mesh->get_faces().empty()) {
      std::cout << "No Faces - skipping Sub-Model " << sub_name << std::endl;
      continue;
    }

    std::cout << "Model includes " << sub_mesh->get_faces().size()/3 << " of "
              << mesh->get_faces().size()/3 << " faces." << std::endl;

    // redo_patches
    // for n-channel create parallel structure here and output a class obj file too
    tex::TexturePatches sub_texture_patches {};
    tex::TexturePatches sub_texture_object_class_patches {};
    int patch_ct = 0;
    for(std::size_t i = 0; i < texture_patches.size(); ++i) {
      TexturePatch::Ptr new_patch = TexturePatch::create(texture_patches[i], face_indices);
      TexturePatch::Ptr new_object_class_patch = nullptr;
      if (texture_channels > num_colors) {
        new_object_class_patch = TexturePatch::create(texture_object_class_patches[i], face_indices);
      }
      if (!new_patch->get_faces().empty()) {
        new_patch->set_label(patch_ct);
        sub_texture_patches.emplace_back(std::move(new_patch));
        if (texture_channels > num_colors) {
          new_object_class_patch->set_label(patch_ct);
          sub_texture_object_class_patches.emplace_back(std::move(new_object_class_patch));
        }
        patch_ct++;
      }
    }

    std::cout << "\tModel took: " << modeltimer.get_elapsed_sec() << "s" << std::endl;
    if (texture_patches.empty()) {
      std::cout << "No Texture Patches - skipping Sub-Model " << sub_name << std::endl;
      continue;
    }
    std::cout << sub_texture_patches.size() << " of "
              << texture_patches.size() << " patches." << std::endl;

    util::WallTimer atlastimer {};
    {
      /* Generate texture atlases. */
      std::cout << "Generating texture atlases:" << std::endl;
      tex::generate_texture_atlases(&sub_texture_patches,
                                    settings,
                                    &sub_texture_atlases,
                                    mesh->get_vertices(),
                                    mesh->get_faces());
    }

    /* Create and write out obj model. */
    {
      std::cout << "Building objmodel:" << std::endl;
      tex::Model sub_model {};
      tex::build_model(sub_mesh, sub_texture_atlases, &sub_model);

      std::cout << "\tSaving model to " << out_prefix+sub_name << "... " << std::flush;
      tex::Model::save(sub_model, out_prefix+sub_name);
      std::cout << "done." << std::endl;
    }
    std::cout << "\tAtlas took: " << atlastimer.get_elapsed_sec() << "s" << std::endl;

    if (texture_channels > num_colors) {
      util::WallTimer classatlastimer {};
      {
        /* Generate texture atlases for object classes. */
        std::cout << "Generating object class texture atlases:" << std::endl;
        tex::generate_texture_atlases(&sub_texture_object_class_patches,
            settings,
            &sub_texture_object_class_atlases,
            mesh->get_vertices(),
            mesh->get_faces());
      }

      /* Create and write out obj model for object classes. */
      {
        std::cout << "Building object class objmodel:" << std::endl;
        tex::Model sub_model {};
        tex::build_model(sub_mesh, sub_texture_object_class_atlases, &sub_model);

        std::cout << "\tSaving object class model to " << out_prefix+sub_name << "_classes... " << std::flush;
        tex::Model::save(sub_model, out_prefix+sub_name+"_classes");
        std::cout << "done." << std::endl;
      }
      std::cout << "\tClass atlas took: " << classatlastimer.get_elapsed_sec() << "s" << std::endl;
    }
  }

  std::cout << "Done" << std::endl;
  std::cout << "\tTotal time: " << timer.get_elapsed_sec() << "s" << std::endl;
}
