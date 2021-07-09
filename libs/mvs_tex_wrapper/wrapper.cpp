#include "mvs_tex_wrapper/wrapper.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mve/mesh_io_ply.h>
#include <util/file_system.h>
#include <util/system.h>
#include <util/timer.h>

#include "tex/debug.h"
#include "tex/progress_counter.h"
#include "tex/settings.h"
#include "tex/texture_patch.h"
#include "tex/texturing.h"
#include "tex/timer.h"
#include "tex/util.h"

namespace MvsTexturing {

using ::std::string;
using ::std::vector;

void textureMesh(
    const TextureSettings& texture_settings,
    const string& in_scene,
    const string& in_mesh,
    const string& out_prefix,
    const vector<vector<bool>>& sub_vert_masks,
    const vector<string>& sub_names,
    std::shared_ptr<EuclideanViewMask> ev_mask,
    uint atlas_size,
    float* hidden_face_proportion,
    std::shared_ptr<std::vector<std::vector<uint8_t>>> segmentation_classes,
    std::shared_ptr<std::vector<std::vector<uint8_t>>> texture_atlas_colors) {
  bool write_intermediate_results = false;
  bool do_texture_atlas = true;
  if (segmentation_classes) {
    if (!texture_atlas_colors) {
      do_texture_atlas = false;
    }
  }
  // the number of channels in the image
  int num_texture_channels = 0;
  // the number of image channels that describe color -- additional channels
  // indicate segmentation classes
  // TODO dwh: get rid of hard coded color channels=3
  int num_colors = 3;

  if (atlas_size == 0) {
    atlas_size = 16384;
    // atlas_size = 8192;
    // atlas_size = 4096;
  }
//  atlas_size = MAX_TEXTURE_SIZE;

  std::cout << "Texturing ...\n Eigen version:" << std::endl;
  std::cout << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
            << EIGEN_MINOR_VERSION << std::endl;

  string data_cost_file {};
  string labeling_file {};

  Timer timer {};
  util::WallTimer wtimer {};

  //
  // Prep Filesystem + load data
  //

  string const out_dir = util::fs::dirname(out_prefix);

  if (!util::fs::dir_exists(out_dir.c_str())) {
    std::cerr << "Destination directory does not exist!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  string const tmp_dir = util::fs::join_path(out_dir, "tmp");

  if (!util::fs::dir_exists(tmp_dir.c_str())) {
    util::fs::mkdir(tmp_dir.c_str());
  }

  std::cout << "Load and prepare mesh: " << in_mesh << std::endl;

  mve::TriangleMesh::Ptr mesh {};

  try {
    mesh = mve::geom::load_ply_mesh(in_mesh);
  }

  catch (std::exception& e) {
    std::cerr << "\tCould not load mesh: " << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  mve::MeshInfo mesh_info(mesh);
  tex::prepare_mesh(&mesh_info, mesh);

  std::cout << "Generating texture views: " << std::endl;

  tex::TextureViews texture_views {};

  tex::generate_texture_views(in_scene, &texture_views, tmp_dir);

  if (!texture_views.empty()) {
    num_texture_channels = texture_views[0].get_channels();
  }

  timer.measure("Loading");

  std::size_t const num_faces = mesh->get_faces().size() / 3;

  std::cout << "Building adjacency graph: " << std::endl;

  tex::Graph graph(num_faces);

  tex::build_adjacency_graph(mesh, mesh_info, &graph);

  //
  // Build Processing Settings
  //
  tex::Settings settings {};

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
  settings.hole_filling = texture_settings.do_hole_filling;
  settings.keep_unseen_faces = texture_settings.do_keep_unseen_faces;
  
  settings.dilate_padding_pixels = texture_settings.do_dilate_padding_pixels;
  settings.highlight_padding_pixels = texture_settings.do_highlight_padding_pixels;
  settings.expose_blending_mask = texture_settings.do_expose_blending_mask;
  settings.expose_validity_mask = texture_settings.do_expose_validity_mask;
  settings.scale_if_needed = texture_settings.do_scale_if_needed;

  settings.texture_scaling_adj = texture_settings.texture_scaling_adj;
  settings.texture_scaling_backstop = texture_settings.texture_scaling_backstop;
  settings.texture_scaling_min = texture_settings.texture_scaling_min;
  settings.texture_scaling_max_iterations = texture_settings.texture_scaling_max_iterations;
  
  std::cout << "dilate_padding_pixels: " << settings.dilate_padding_pixels << std::endl;
  std::cout << "highlight_padding_pixels: " << settings.highlight_padding_pixels << std::endl;
  std::cout << "expose_blending_mask: " << settings.expose_blending_mask << std::endl;
  std::cout << "expose_validity_mask: " << settings.expose_validity_mask << std::endl;
  std::cout << "scale_if_needed: " << settings.scale_if_needed << std::endl;

  if (labeling_file.empty()) {
    std::cout << "View selection:" << std::endl;

    util::WallTimer rwtimer {};

    tex::DataCosts data_costs(static_cast<uint32_t>(num_faces),
      static_cast<uint16_t>(texture_views.size()));

    if (data_cost_file.empty()) {
      // std::cout << "- added - Calculating Data costs" << std::endl;
      tex::calculate_data_costs(
          mesh,
          &texture_views,
          settings,
          &data_costs,
          ev_mask,
          hidden_face_proportion);

      if (write_intermediate_results) {
        std::cout << "\tWriting data cost file... " << std::flush;

        tex::DataCosts::save_to_file(
            data_costs, out_prefix + "_data_costs.spt");

        std::cout << "done." << std::endl;
      }
    } else {
      std::cout << "\tLoading data cost file... " << std::flush;

      try {
        tex::DataCosts::load_from_file(data_cost_file, &data_costs);
      }

      catch (util::FileException& e) {
        std::cout << "failed!" << std::endl;
        std::cerr << e.what() << std::endl;

        std::exit(EXIT_FAILURE);
      }

      std::cout << "done." << std::endl;
    }

    timer.measure("Calculating data costs");

    std::cout << "- added - Selecting Views" << std::endl;

    try {
      tex::view_selection(data_costs, &graph, settings);
    }

    catch (std::runtime_error& e) {
      std::cerr << "\tOptimization failed: " << e.what() << std::endl;

      std::exit(EXIT_FAILURE);
    }

    timer.measure("Running MRF optimization");

    std::cout << "\tTook: " << rwtimer.get_elapsed_sec() << "s" << std::endl;

    /* Write labeling to file. */
    if (write_intermediate_results) {
      vector<std::size_t> labeling(graph.num_nodes());

      for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        labeling[i] = graph.get_label(i);
      }

      vector_to_file(out_prefix + "_labeling.vec", labeling);
    }
  } else {
    std::cout << "Loading labeling from file... " << std::flush;

    /* Load labeling from file. */
    vector<std::size_t> labeling = vector_from_file<std::size_t>(labeling_file);

    if (labeling.size() != graph.num_nodes()) {
      std::cerr
          << "Wrong labeling file for this mesh/scene combination... aborting!"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    /* Transfer labeling to graph. */
    for (std::size_t i = 0; i < labeling.size(); ++i) {
      const std::size_t label = labeling[i];

      if (label > texture_views.size()) {
        std::cerr << "Wrong labeling file for this mesh/scene combination... "
                     "aborting!"
                  << std::endl;

        std::exit(EXIT_FAILURE);
      }

      graph.set_label(i, label);
    }

    std::cout << "done." << std::endl;
  }

  //  Generate full texture patches.
  tex::TexturePatches texture_patches {};
  tex::TexturePatches texture_object_class_patches {};

  //  Create texture patches and adjust them.
  tex::VertexProjectionInfos vertex_projection_infos {};

  std::cout << "Generating texture patches:" << std::endl;

  tex::generate_texture_patches(
      graph,
      mesh,
      mesh_info,
      &texture_views,
      settings,
      &vertex_projection_infos,
      &texture_patches);

  if (settings.global_seam_leveling) {
    //  FIXME - bitweeder
    //  Making global seam leveling work with scaling will probably require
    //  calling adjust_colors from within global_seam_leveling with
    //  only_regenerate_masks set to true when scaling, then relying on the
    //  post-scaling call to adjust_colors to do the actual adjustment.
    //  This theory is untested.
    
    // TODO dwh: can we get here if doing segmentation??
    std::cout << "Running global seam leveling:" << std::endl;

    tex::global_seam_leveling(
        graph, mesh, mesh_info, vertex_projection_infos, &texture_patches);

    timer.measure("Running global seam leveling");
  } else {
    ProgressCounter texture_patch_counter(
        "Calculating validity masks for texture patches",
        texture_patches.size());

    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < texture_patches.size(); ++i) {
      texture_patch_counter.progress<SIMPLE>();

      auto texture_patch = texture_patches[i];
      
      //  FIXME - bitweeder
      //  This is a wasted allocation if we only care about generating masks.
      vector<math::Vec3f> patch_adjust_values(
          texture_patch->get_faces().size() * 3, math::Vec3f(0.0f));

     if (settings.scale_if_needed) {
        //  We’ll be running adjust_colors after scaling, in this case; running
        //  it twice would lead to artifacts. We do, however, still need the
        //  masks generated for local seam leveling.
        texture_patch->adjust_colors(patch_adjust_values, true);
      } else {
        texture_patch->adjust_colors(patch_adjust_values);
      }
      
      texture_patch_counter.inc();
    }

    timer.measure("Calculating texture patch validity masks");
  }

  if (num_texture_channels > num_colors) {
    std::cout << "Making object class textures:" << std::endl;

    // Build a copy of texture_patches for object classes
    for (const auto& texture_patch : texture_patches) {
      texture_object_class_patches.emplace_back(texture_patch->duplicate());
    }

    if (do_texture_atlas && texture_atlas_colors) {
      // we need this if doing an obj texture atlas
      // This method creates a synthetic color image where each rgb color
      // represents a different class
      std::cout << "Building object class texture image:" << std::endl;

      #pragma omp parallel for schedule(dynamic)
      for (std::size_t i = 0; i < texture_object_class_patches.size(); ++i) {
        auto texture_object_class_patch = texture_object_class_patches[i];
        vector<math::Vec3f> patch_adjust_values(
            texture_object_class_patch->get_faces().size() * 3,
            math::Vec3f(0.0f));

        texture_object_class_patch->adjust_colors(
            patch_adjust_values, false, num_texture_channels, texture_atlas_colors);
      }
    }

    // For n-channel, do the following on all channels including classes
    if (settings.local_seam_leveling) {
      if (do_texture_atlas && texture_atlas_colors) {
        std::cout << "Running local seam leveling with classes:" << std::endl;

        // This function call does seam leveling on everything including rgb
        // class data
        tex::local_seam_leveling_n(
            graph,
            mesh,
            vertex_projection_infos,
            &texture_patches,
            num_texture_channels,
            &texture_object_class_patches,
            texture_atlas_colors);
      } else {
        // if we are not outputting an obj texture atlas file the following is
        // a better option than the above: This function call ignores rgb
        // object segmentation class data while doing seam leveling and just
        // does rgb channels
        std::cout << "Running local seam leveling ignoring object classes:"
                  << std::endl;

        tex::local_seam_leveling_n(
            graph,
            mesh,
            vertex_projection_infos,
            &texture_patches,
            num_texture_channels,
            nullptr,
            texture_atlas_colors);
      }
    }

    timer.measure("Running local seam leveling with object classes");

    if (segmentation_classes) {
      std::cout << "Setting segmentation class probabilities for "
                << vertex_projection_infos.size()
                << " vertices:" << std::endl;

      segmentation_classes->clear();

      // set the segmentation class for each vertex
      for (std::size_t i = 0; i < vertex_projection_infos.size(); ++i) {
        vector<tex::VertexProjectionInfo> const& projection_infos =
            vertex_projection_infos[i];
        vector<float> texture_channels(num_texture_channels);
        int number_projections = 0;

        for (auto const& projection_info : projection_infos) {
          auto texture_patch = texture_patches.at(projection_info.texture_patch_id);

          if (texture_patch->get_label() == 0) {
            continue;
          }

          number_projections += 1;

          auto pixel_channels = texture_patch->get_pixel_value_n(
              projection_info.projection, num_texture_channels);

          std::transform(
              texture_channels.begin(),
              texture_channels.end(),
              pixel_channels.begin(),
              texture_channels.begin(),
              std::plus<float>());
        }

        float normalize_factor =
            (number_projections > 0)
                ? 255.f / static_cast<float>(number_projections)
                : 255.f;

        for (auto&& channel : texture_channels) {
          channel *= normalize_factor;
        }

        segmentation_classes->emplace_back(
            texture_channels.begin() + num_colors, texture_channels.end());
      }

      timer.measure("Creating object class assignments");
    }
  } else {
    if (settings.local_seam_leveling) {
      std::cout << "Running local seam leveling:" << std::endl;

      tex::local_seam_leveling(
          graph, mesh, vertex_projection_infos, &texture_patches);
    }

    timer.measure("Running local seam leveling");

    // ensure that we always do texture atlas if we are not doing segmentation
    do_texture_atlas = true;
  }

  //  do this--otherwise skip to cleanup and exit
  if (do_texture_atlas) {
    //  FIXME - bitweeder
    //  This OpenMP directive is currently disabled in order to avoid an
    //  intermittent failure when tiling meshes. Apparently, the scaling code
    //  in generate_capped_texture_atlas broke concurrency, and we occasionally
    //  end up in a race that can crash the module. Running this loop serially
    //  avoids the issue, though it’s not a great workaround (there is small,
    //  but measurable time penalty we incur, now). I’m leaving it this way
    //  for expediency, but realisticaly, we’ll probably replace mvs-texturing
    //  before we spend more time patching it.
    
    //  Now loop, generating+saving subindexed meshes and atlas
//    #pragma omp parallel for schedule(dynamic)
    for (std::size_t vi = 0; vi < sub_vert_masks.size(); ++vi) {
      std::cout << "\nFinalizing Sub-Model " << sub_names[vi] << " - " << vi + 1
                << " of " << sub_vert_masks.size() << std::endl;

      tex::TextureAtlases sub_texture_atlases {};
      tex::TextureAtlases sub_texture_object_class_atlases {};
      const vector<bool>& vertex_mask {sub_vert_masks[vi]};
      vector<bool> inverted_mask(vertex_mask.size());

      for (std::size_t i = 0; i < vertex_mask.size(); ++i) {
        inverted_mask[i] = !vertex_mask[i];
      }

      const string& sub_name {sub_names[vi]};
      vector<std::size_t> face_indices {};

      // generate face reindex
      generate_face_reindex(vertex_mask, mesh->get_faces(), face_indices);

      // redo mesh
      mve::TriangleMesh::Ptr sub_mesh = mesh->duplicate();

      sub_mesh->delete_vertices_fix_faces(inverted_mask);

      if (sub_mesh->get_faces().empty()) {
        std::cout << "No Faces - skipping Sub-Model " << sub_name << std::endl;
        continue;
      }

      std::cout << "Model includes " << sub_mesh->get_faces().size() / 3
                << " of " << mesh->get_faces().size() / 3 << " faces."
                << std::endl;

      // redo_patches
      tex::TexturePatches sub_texture_patches {};
      tex::TexturePatches sub_texture_object_class_patches {};
      size_t patch_ct = 0;

      for (std::size_t i = 0; i < texture_patches.size(); ++i) {
        auto new_patch = TexturePatch::create(texture_patches[i], face_indices);
        TexturePatch::Ptr new_object_class_patch = nullptr;

        if (num_texture_channels > num_colors) {
          new_object_class_patch = TexturePatch::create(
              texture_object_class_patches[i], face_indices);
        }

        if (!new_patch->get_faces().empty()) {
          new_patch->set_label(static_cast<int>(patch_ct));
          sub_texture_patches.emplace_back(std::move(new_patch));

          if (num_texture_channels > num_colors) {
            new_object_class_patch->set_label(static_cast<int>(patch_ct));
            sub_texture_object_class_patches.emplace_back(
                std::move(new_object_class_patch));
          }

          patch_ct++;
        }
      }

      if (texture_patches.empty()) {
        std::cout << "No Texture Patches - skipping Sub-Model " << sub_name
                  << std::endl;

        continue;
      }

      std::cout << sub_texture_patches.size() << " of "
                << texture_patches.size() << " patches." << std::endl;

      //  Generate texture atlases.
      std::cout << "\nGenerating texture atlases: " << std::flush;
      if (settings.scale_if_needed) {
        tex::generate_capped_texture_atlas(
          &sub_texture_patches,
          settings,
          &sub_texture_atlases,
          static_cast<uint>(atlas_size),
          mesh->get_vertices(),
          mesh->get_faces());
      } else {
        tex::generate_texture_atlases(
            &sub_texture_patches,
            settings,
            &sub_texture_atlases,
            mesh->get_vertices(),
            mesh->get_faces());
      }

      //  Create and write out obj model.
      {
        std::cout << "Building objmodel:" << std::endl;

        tex::Model sub_model {};

        tex::build_model(sub_mesh, sub_texture_atlases, &sub_model);
        timer.measure("Building OBJ model");

        std::cout << "\tSaving model to " << out_prefix + sub_name << "... "
                  << "done." << std::endl;

        tex::Model::save(sub_model, out_prefix + sub_name);
        timer.measure("Saving");
      }

      if (num_texture_channels > num_colors && texture_atlas_colors) {
        // only do this if doing segmentation and have a texture color vector
        {
          //  Generate texture atlases for object classes.
          std::cout << "Generating object class texture atlases:" << std::endl;

          tex::generate_texture_atlases(
              &sub_texture_object_class_patches,
              settings,
              &sub_texture_object_class_atlases,
              mesh->get_vertices(),
              mesh->get_faces());
        }

        //  Create and write out obj model for object classes.
        {
          std::cout << "Building object class objmodel:" << std::endl;

          tex::Model sub_model {};

          tex::build_model(
              sub_mesh, sub_texture_object_class_atlases, &sub_model);
          timer.measure("Building OBJ class model");

          std::cout << "\tSaving object class model to "
                    << out_prefix + sub_name << "_classes... " << std::flush;
          std::cout << "done." << std::endl;

          tex::Model::save(sub_model, out_prefix + sub_name + "_classes");
          timer.measure("Saving object model");
        }
      }

      timer.measure("Total");
    }
  }
  
  std::cout << "Whole texturing procedure took: " << wtimer.get_elapsed_sec()
            << "s" << std::endl;

  //  Remove temporary files
  for (util::fs::File const& file : util::fs::Directory(tmp_dir)) {
    util::fs::unlink(util::fs::join_path(file.path, file.name).c_str());
  }
  
  util::fs::rmdir(tmp_dir.c_str());
}

void generate_vertex_reindex(
    const vector<bool>& mask,
    vector<std::size_t>& new_indices) {
  new_indices.resize(mask.size());

  std::size_t ct = 0;

  for (std::size_t i = 0; i < new_indices.size(); ++i) {
    if (mask[i]) {
      new_indices[i] = ct;
      ++ct;
    } else {
      new_indices[i] = std::numeric_limits<std::size_t>::max();
    }
  }
}

bool is_valid_tri(
    std::size_t i,
    const vector<bool>& mask,
    const vector<unsigned int>& old_faces);

bool is_valid_tri(
    std::size_t i,
    const vector<bool>& mask,
    const vector<unsigned int>& old_faces) {
  return mask[old_faces[i * 3]] && mask[old_faces[i * 3 + 1]]
         && mask[old_faces[i * 3 + 2]];
}

/**
  @brief Strange reindexing to match the swap-based MVE reduction
*/
void generate_face_reindex(
    const vector<bool>& mask,
    const vector<unsigned int>& old_faces,
    vector<std::size_t>& new_indices) {
  new_indices.resize(old_faces.size() / 3);
  std::size_t front = 0;
  std::size_t back = new_indices.size() - 1;
  while (front < back) {
    if (is_valid_tri(front, mask, old_faces)) {
      new_indices[front] = front;
      ++front;
    } else {
      while (front < back && !is_valid_tri(back, mask, old_faces)) {
        new_indices[back] = std::numeric_limits<std::size_t>::max();
        --back;
      }
      
      new_indices[front] = std::numeric_limits<std::size_t>::max();

      if (is_valid_tri(back, mask, old_faces)) {
        // note - front may equal back here, but the desired behavior will still
        // happen.
        new_indices[back] = front;
        back--;
        front++;
      }
    }
  }
}

}  // namespace MvsTexturing
