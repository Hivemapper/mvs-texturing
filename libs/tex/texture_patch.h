/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_TEXTUREPATCH_HEADER
#define TEX_TEXTUREPATCH_HEADER

#include <vector>

#include <math/vector.h>
#include <mve/mesh.h>

#include "poisson_blending.h"
#include "tri.h"

int const texture_patch_border = 1;

/**
 * Class representing a texture patch.
 * Contains additionally to the rectangular part of the TextureView
 * the faces which it textures and their relative texture coordinates.
 */
class TexturePatch {
public:
  typedef std::shared_ptr<TexturePatch> Ptr;
  typedef std::vector<std::size_t> Faces;
  typedef std::vector<math::Vec2f> Texcoords;

private:
  int label;
  Faces faces;
  Texcoords texcoords;
  mve::FloatImage::Ptr image {};
  mve::ByteImage::Ptr validity_mask {};
  mve::ByteImage::Ptr blending_mask {};

public:
  /** Constructs a texture patch. */
  TexturePatch(
      int _label,
      std::vector<std::size_t> const& _faces,
      std::vector<math::Vec2f> const& _texcoords,
      mve::FloatImage::Ptr _image);

  TexturePatch(TexturePatch const& texture_patch);

  TexturePatch(
      TexturePatch const& texture_patch,
      const std::vector<std::size_t>& new_face_indices);

  static TexturePatch::Ptr create(TexturePatch::Ptr texture_patch);

  static TexturePatch::Ptr create(
      TexturePatch::Ptr texture_patch,
      const std::vector<std::size_t>& new_face_indices);

  static TexturePatch::Ptr create(
      int label,
      std::vector<std::size_t> const& faces,
      std::vector<math::Vec2f> const& texcoords,
      mve::FloatImage::Ptr image);

  TexturePatch::Ptr duplicate();

  void rescale(double ratio);

  /** Adjust the image colors and update validity mask. */
  void adjust_colors(
      std::vector<math::Vec3f> const& adjust_values,
      int num_channels = 3);

  math::Vec3f get_pixel_value(math::Vec2f pixel) const;
  std::vector<float> get_pixel_value_n(math::Vec2f pixel, int num_channels)
      const;

  void set_pixel_value(math::Vec2i pixel, math::Vec3f color);
  void set_pixel_value(math::Vec2i pixel, const std::vector<float>* color);
  void set_pixel_object_class_value(
      math::Vec2i pixel,
      const std::vector<float>* color);
  static math::Vec3f compute_object_class_color(
      const std::vector<float>* color);

  bool valid_pixel(math::Vec2i pixel) const;
  bool valid_pixel(math::Vec2f pixel) const;

  std::vector<std::size_t>& get_faces();
  std::vector<std::size_t> const& get_faces() const;
  std::vector<math::Vec2f>& get_texcoords();
  std::vector<math::Vec2f> const& get_texcoords() const;

  mve::FloatImage::Ptr get_image();

  mve::FloatImage::ConstPtr get_image() const;
  mve::ByteImage::ConstPtr get_validity_mask() const;
  mve::ByteImage::ConstPtr get_blending_mask() const;

  //        std::pair<float, float> get_min_max(void) const;

  void release_blending_mask();
  void prepare_blending_mask(std::size_t strip_width);

  //        void erode_validity_mask();

  void blend(mve::FloatImage::ConstPtr orig);

  int get_label() const;
  void set_label(int l) {
    label = l;
  }
  int get_width() const;
  int get_height() const;
  int get_channels() const;
  int get_size() const;

  double compute_geometric_area(
      const std::vector<math::Vec3f>& vertices,
      const std::vector<uint>& mesh_faces) const;
  double compute_pixel_area() const;
};

inline TexturePatch::Ptr TexturePatch::create(
    TexturePatch::Ptr texture_patch) {
  return std::make_shared<TexturePatch>(*texture_patch);
}

inline TexturePatch::Ptr TexturePatch::create(
    TexturePatch::Ptr texture_patch,
    const std::vector<std::size_t>& new_face_indices) {
  return std::make_shared<TexturePatch>(*texture_patch, new_face_indices);
}

inline TexturePatch::Ptr TexturePatch::create(
    int label,
    std::vector<std::size_t> const& faces,
    std::vector<math::Vec2f> const& texcoords,
    mve::FloatImage::Ptr image) {
  return std::make_shared<TexturePatch>(label, faces, texcoords, image);
}

inline TexturePatch::Ptr TexturePatch::duplicate() {
  return Ptr(new TexturePatch(*this));
}

inline int TexturePatch::get_label() const {
  return label;
}

inline int TexturePatch::get_width() const {
  return image->width();
}

inline int TexturePatch::get_height() const {
  return image->height();
}

inline int TexturePatch::get_channels() const {
  return image->channels();
}

inline mve::FloatImage::Ptr TexturePatch::get_image() {
  return image;
}

inline mve::FloatImage::ConstPtr TexturePatch::get_image() const {
  return image;
}

inline mve::ByteImage::ConstPtr TexturePatch::get_validity_mask() const {
  return validity_mask;
}

inline mve::ByteImage::ConstPtr TexturePatch::get_blending_mask() const {
  assert(blending_mask != nullptr);
  return blending_mask;
}

inline void TexturePatch::release_blending_mask() {
  assert(blending_mask != nullptr);
  blending_mask.reset();
}

inline std::vector<math::Vec2f>& TexturePatch::get_texcoords() {
  return texcoords;
}

inline std::vector<std::size_t>& TexturePatch::get_faces() {
  return faces;
}

inline std::vector<math::Vec2f> const& TexturePatch::get_texcoords() const {
  return texcoords;
}

inline std::vector<std::size_t> const& TexturePatch::get_faces() const {
  return faces;
}

inline int TexturePatch::get_size() const {
  return get_width() * get_height();
}

#endif /* TEX_TEXTUREPATCH_HEADER */
