#pragma once

#include <memory>
#include <string>

#include <mvs_tex_mask/euclidean_view_mask.h>

namespace MvsTexturing {

struct TextureSettings {
    bool do_use_gmi_term = false;  // GMI vs area
    bool do_gauss_clamping = true;  // one or other true, not both. Photmetric consistency type
    bool do_gauss_damping = false;
    bool do_geometric_visibility_test = false;
    bool do_gamma_tone_mapping = true;
    bool do_global_seam_leveling = false;
    bool do_local_seam_leveling = true;
    bool do_hole_filling = true;
    bool do_keep_unseen_faces = true;
};

void textureMesh(const TextureSettings& texture_settings,
                 const std::string& in_scene,
                 const std::string& in_mesh,
                 const std::string& out_prefix,
                 std::shared_ptr<EuclideanViewMask> ev_mask = NULL);
}  // namespace MvsTexturing
