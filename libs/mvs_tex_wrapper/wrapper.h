#pragma once

#include <memory>
#include <string>
#include <vector>

#include <mvs_tex_mask/euclidean_view_mask.h>

namespace MvsTexturing {

struct TextureSettings {
  //  This is the scaling applied to our texture atlas candidate charts on
  //  every iteration of the page-fitting algorithm. It should be conservative,
  //  as the algorithm is already careful about page-fitting for reasonable
  //  candidates (convergence within 2-3 iterations for 98%+ of candidates).
  double texture_scaling_adj = 0.99;

  //  This is the last ditch value attempted after the nth scaing iteration. We
  //  are, at this point dealing with a pathological case, and so the goal is
  //  to ensure that a representative atlas page is generated _at all_. As
  //  such, any sufficiently small scaling will do, though we’d prefer
  //  something that doesn’t result in a single color representng the whole
  //  mesh tile. `0.666` was used successully with sub-million-point meshes of
  //  consistent density.
  double texture_scaling_backstop = 0.666;

  //  This represents the smallest scaling we allow relative to the original
  //  computed texture atlas chart sizes. Long before we get anywhere near
  //  this, we’ll be experiencing a severe loss of quality relative to the
  //  expected level of detail. If we would need to shrink the atlas more than
  //  this, we assume the atlas is not worth preserving.
  //
  //  SEEME - bitweeder
  //  Arguably, we could be much more liberal with what we accept here, as the
  //  penalty for not having an atlas is severe (i.e., a missing mesh tile on
  //  the map).
  double texture_scaling_min = 0.01;

  //  This is the largest number of iterations we perform in the texture atlas
  //  scaling algorithm before giving up. This is intended as a defense against
  //  pathological cases, as testing has shown we generally converge very, very
  //  quickly.
  std::size_t texture_scaling_max_iterations = 10;

  bool do_use_gmi_term = false;  // GMI vs area
  bool do_gauss_clamping = true; // one or other true, not both. Photometric consistency type
  bool do_gauss_damping = false;
  bool do_geometric_visibility_test = false;
  bool do_gamma_tone_mapping = true;
  bool do_global_seam_leveling = false;
  bool do_local_seam_leveling = true;
  bool do_hole_filling = true;
  bool do_keep_unseen_faces = true;
  
  bool do_dilate_padding_pixels = false;
  bool do_highlight_padding_pixels = false;
  bool do_expose_blending_mask = false;
  bool do_expose_validity_mask = false;
  bool do_scale_if_needed = false;
};

void generate_vertex_reindex(
    const std::vector<bool>& mask,
    std::vector<std::size_t>& new_indices);

void generate_face_reindex(
    const std::vector<bool>& mask,
    const std::vector<unsigned int>& old_faces,
    std::vector<std::size_t>& new_indices);

void textureMesh(
    const TextureSettings& texture_settings,
    const std::string& in_scene,
    const std::string& in_mesh,
    const std::string& out_prefix,
    const std::vector<std::vector<bool>>& sub_vert_masks,
    const std::vector<std::string>& sub_names,
    std::shared_ptr<EuclideanViewMask> ev_mask,
    uint atlas_size,
    float* hidden_face_proportion,
    std::shared_ptr<std::vector<std::vector<uint8_t>>> segmentation_classes,
    std::shared_ptr<std::vector<std::vector<uint8_t>>> texture_atlas_colors = nullptr);
//  if segmentation classes are to be set (i.e. not a nullptr),
//  then setting this to false stops method after setting
//  segmentation classes to avoid wasting time if textures are not
//  needed. setting to false only has an effect if image has more
//  channels than colors--otherwise defaults to doing the atlas
//  every time

}  // namespace MvsTexturing
