#pragma once

#include <memory>
#include <string>
#include <vector>

#include <mvs_tex_mask/euclidean_view_mask.h>

namespace MvsTexturing {

struct TextureSettings {
  bool do_use_gmi_term = false; // GMI vs area
  bool do_gauss_clamping =
      true; // one or other true, not both. Photmetric consistency type
  bool do_gauss_damping = false;
  bool do_geometric_visibility_test = false;
  bool do_gamma_tone_mapping = true;
  bool do_global_seam_leveling = false;
  bool do_local_seam_leveling = true;
  bool do_hole_filling = true;
  bool do_keep_unseen_faces = true;
};

void generate_vertex_reindex(const std::vector<bool> &mask,
                             std::vector<std::size_t> &new_indices);

void generate_face_reindex(const std::vector<bool> &mask,
                           const std::vector<unsigned int> &old_faces,
                           std::vector<std::size_t> &new_indices);

void textureMesh(
    const TextureSettings &texture_settings, const std::string &in_scene,
    const std::string &in_mesh, const std::string &out_prefix,
    const std::vector<std::vector<bool>> &sub_vert_masks,
    const std::vector<std::string> &sub_names,
    std::shared_ptr<EuclideanViewMask> ev_mask = NULL, uint atlas_size = 0,
    float *hidden_face_proportion = NULL,
    std::vector<std::vector<uint8_t>> *segmentation_classes = nullptr,
    bool do_texture_atlas =
        true); // if segmentation classes are to be set (i.e. not a nullptr),
               // then setting this to false stops method after setting
               // segmentation classes to avoid wasting time if textures are not
               // needed.
// setting to false only has an effect if image has more channels than
// colors--otherwise defaults to doing the atlas every time
} // namespace MvsTexturing
