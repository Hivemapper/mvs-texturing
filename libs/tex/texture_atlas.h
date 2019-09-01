/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_TEXTUREATLAS_HEADER
#define TEX_TEXTUREATLAS_HEADER

#include <vector>

#include <math/vector.h>
#include <mve/image.h>
#include <mve/mesh.h>
#include <util/exception.h>

#include "rectangular_bin.h"
#include "settings.h"
#include "texture_patch.h"
#include "tri.h"

/**
 * Class representing a texture atlas.
 */
class TextureAtlas {
public:
  typedef std::shared_ptr<TextureAtlas> Ptr;

  typedef std::vector<std::size_t> Faces;
  typedef std::vector<std::size_t> TexcoordIds;
  typedef std::vector<math::Vec2f> Texcoords;

private:
  unsigned int const size;
  bool finalized {false};

  Faces faces;
  Texcoords texcoords;
  TexcoordIds texcoord_ids;

  mve::ByteImage::Ptr image;
  mve::ByteImage::Ptr validity_mask;

  RectangularBin::Ptr bin;

  void apply_edge_padding(tex::Settings const& settings);
  void merge_texcoords();

public:
  TextureAtlas(unsigned int size);

  static TextureAtlas::Ptr create(unsigned int size);

  Faces const& get_faces() const;
  TexcoordIds const& get_texcoord_ids() const;
  Texcoords const& get_texcoords() const;
  mve::ByteImage::ConstPtr get_image() const;

  uint insert(TexturePatch::Ptr texture_patch);

  void finalize(tex::Settings const& settings);
};

/**
  @brief Calculate the default padding around each chart in the atlas.
  @details The calculated value is based strctly on the edge length of the
  atlas. The actual padding used for a given chart is uses this as a starting
  point, but may vary substantially depending on the characteristics of the
  chart and any usage requirements.
*/
inline uint compute_base_padding(uint edge_length) {
  return std::min(12u, edge_length >> 8);
}

/**
  @brief Return the calculated texel padding required by a give chart.
  @details A given chart is guaranteed to have no fewer than 2 texels and
  no more than max_padding texels of padding surrounding it on the atlas page.
  Within this range, the padding is 1/16 of the larger of the bounding rect’s
  width and height. As implied, the amount of padding is constant in both
  height and width.
 
  @param base_width is the width of the chart’s bounding rect.
  @param base_height is the height of the chart’s bounding rect.
  @param max_padding is the maximum value that may be returned.
 
  @return a positive integer padding value in [2, `max_padding`]
*/
inline uint compute_local_padding(
    uint base_width,
    uint base_height,
    uint edge_length) {
  uint max_padding = compute_base_padding(edge_length);
  uint size = std::max(base_width, base_height);
  uint local_padding = std::min(std::max(2u, size >> 4), max_padding);

  //  SEEME - bitweeder
  //  It’s excessive to have a border wider than 2 pixels, even with
  //  anisotropic/trilinear filtering. The original logic has been preserved,
  //  but theresult is now hard-coded.
  return 2;
//  return local_padding;
}

inline TextureAtlas::Ptr TextureAtlas::create(unsigned int size) {
  return Ptr(new TextureAtlas(size));
}

inline TextureAtlas::Faces const& TextureAtlas::get_faces() const {
  return faces;
}

inline TextureAtlas::TexcoordIds const& TextureAtlas::get_texcoord_ids() const {
  return texcoord_ids;
}

inline TextureAtlas::Texcoords const& TextureAtlas::get_texcoords() const {
  return texcoords;
}

inline mve::ByteImage::ConstPtr TextureAtlas::get_image() const {
  if (!finalized) {
    throw util::Exception("Texture atlas not finalized");
  }
  return image;
}

#endif /* TEX_TEXTUREATLAS_HEADER */
