#pragma once

#include <memory>
#include <string>

#include "mvs_tex/mask/euclidean_view_mask.h"

namespace MvsTexturing {

std::string testFunc(int n);
void textureMesh(const std::string& in_scene, const std::string& in_mesh, const std::string& out_prefix, std::shared_ptr<EuclideanViewMask> ev_mask = NULL);
}  // namespace MvsTexturing
