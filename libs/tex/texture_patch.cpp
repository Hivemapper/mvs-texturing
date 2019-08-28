/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <cmath>
#include <set>

#include <math/functions.h>
#include <mve/image_color.h>
#include <mve/image_tools.h>
#include <mve/mesh_io_ply.h>

#include "texture_patch.h"

TexturePatch::TexturePatch(
    int label,
    std::vector<std::size_t> const& faces,
    std::vector<math::Vec2f> const& texcoords,
    mve::FloatImage::Ptr image)
  : label(label), faces(faces), texcoords(texcoords), image(image) {
  validity_mask = mve::ByteImage::create(get_width(), get_height(), 1);
  validity_mask->fill(255);
  blending_mask = mve::ByteImage::create(get_width(), get_height(), 1);
}

TexturePatch::TexturePatch(TexturePatch const& texture_patch) {
  label = texture_patch.label;
  faces = std::vector<std::size_t>(texture_patch.faces);
  texcoords = std::vector<math::Vec2f>(texture_patch.texcoords);
  image = texture_patch.image->duplicate();
  validity_mask = texture_patch.validity_mask->duplicate();
  if (texture_patch.blending_mask != nullptr) {
    blending_mask = texture_patch.blending_mask->duplicate();
  }
}

TexturePatch::TexturePatch(
    TexturePatch const& texture_patch,
    const std::vector<std::size_t>& new_face_indices) {
  label = texture_patch.label;
  faces.clear();
  texcoords.clear();
  float pre_x_max, pre_y_max, pre_x_min, pre_y_min, post_x_max, post_y_max,
      post_x_min, post_y_min;
  pre_x_max = pre_y_max = post_x_max = post_y_max = 0;
  pre_x_min = pre_y_min = post_x_min = post_y_min = 1e6;
  for (std::size_t i = 0; i < texture_patch.faces.size(); i++) {
    bool valid =
        (new_face_indices[texture_patch.faces[i]]
         != std::numeric_limits<std::size_t>::max());
    if (valid) {
      faces.push_back(new_face_indices[texture_patch.faces[i]]);
      texcoords.push_back(texture_patch.texcoords[i * 3]);
      texcoords.push_back(texture_patch.texcoords[i * 3 + 1]);
      texcoords.push_back(texture_patch.texcoords[i * 3 + 2]);
    }
  }

  bool resize = false;
  if (!texcoords.empty()
      && texcoords.size() != texture_patch.texcoords.size()) {
    for (std::size_t i = 0; i < texture_patch.texcoords.size(); ++i) {
      pre_x_max = std::max(pre_x_max, texture_patch.texcoords[i][0]);
      pre_x_min = std::min(pre_x_min, texture_patch.texcoords[i][0]);
      pre_y_max = std::max(pre_y_max, texture_patch.texcoords[i][1]);
      pre_y_min = std::min(pre_y_min, texture_patch.texcoords[i][1]);
    }
    for (std::size_t i = 0; i < texcoords.size(); ++i) {
      post_x_max = std::max(post_x_max, texcoords[i][0]);
      post_x_min = std::min(post_x_min, texcoords[i][0]);
      post_y_max = std::max(post_y_max, texcoords[i][1]);
      post_y_min = std::min(post_y_min, texcoords[i][1]);
    }
    float threshold = 5.0;
    if (pre_x_max - post_x_max > threshold || pre_y_max - post_y_max > threshold
        || post_x_min - pre_x_min > threshold
        || post_y_min - pre_y_min > threshold) {
      resize = true;
    }
  }

  if (!resize) {
    image = texture_patch.image->duplicate();
    validity_mask = texture_patch.validity_mask->duplicate();
    if (texture_patch.blending_mask != nullptr) {
      blending_mask = texture_patch.blending_mask->duplicate();
    }
  } else {
    int new_x_start = std::max(0, (int)std::floor(post_x_min - 2));
    int new_y_start = std::max(0, (int)std::floor(post_y_min - 2));
    int new_x_width =
        std::min(texture_patch.image->width(), (int)std::ceil(post_x_max + 2))
        - new_x_start;
    int new_y_width =
        std::min(texture_patch.image->height(), (int)std::ceil(post_y_max + 2))
        - new_y_start;
    int n_im_channels = texture_patch.image->channels();
    int n_vm_channels = texture_patch.validity_mask->channels();
    image = mve::FloatImage::create(new_x_width, new_y_width, n_im_channels);

    validity_mask =
        mve::ByteImage::create(new_x_width, new_y_width, n_vm_channels);
    for (int ci = 0; ci < n_im_channels; ++ci) {
      for (int yi = 0; yi < new_y_width; ++yi) {
        for (int xi = 0; xi < new_x_width; ++xi) {
          image->at(xi, yi, ci) =
              texture_patch.image->at(xi + new_x_start, yi + new_y_start, ci);
        }
      }
    }
    for (int ci = 0; ci < n_vm_channels; ++ci) {
      for (int yi = 0; yi < new_y_width; ++yi) {
        for (int xi = 0; xi < new_x_width; ++xi) {
          validity_mask->at(xi, yi, ci) = texture_patch.validity_mask->at(
              xi + new_x_start, yi + new_y_start, ci);
        }
      }
    }
    for (auto& coord : texcoords) {
      coord[0] -= new_x_start;
      coord[1] -= new_y_start;
    }
  }
}

/**
  @details This coordinate generator mirrors the logic used to scale the
  patches.
*/
math::Vec2f scale_texcoord(
    math::Vec2f const& tc,
    int old_width,
    int old_height,
    int new_width,
    int new_height);
math::Vec2f scale_texcoord(
    math::Vec2f const& tc,
    int old_width,
    int old_height,
    int new_width,
    int new_height) {
  math::Vec2f nrv {};
  
  const float w0 = static_cast<float>(old_width);
  const float w1 = static_cast<float>(new_width - 2.0f * texture_patch_border);
  const float h0 = static_cast<float>(old_height);
  const float h1 = static_cast<float>(new_height - 2.0f * texture_patch_border);

  float x = tc[0] * (w1 / w0) + texture_patch_border;
  float y = tc[1] * (h1 / h0) + texture_patch_border;
  float x_prop = std::min(1.0f, (std::floor(x) + 1 - x) * (w0 / w1));
  float y_prop = std::min(1.0f, (std::floor(y) + 1 - y) * (h0 / h1));

   #define HM_ASSERT(test_) \
      if (!(test_)) {\
        std::cout << "failed: " << #test_ << std::endl; \
        std::cout \
        << "texcoords: (" \
        << tc[0] << ", " \
        << tc[1] << "), " \
        << "old: (" \
        << w0 << ", " \
        << h0 << "), " \
        << "new: (" \
        << w1 << ", " \
        << h1 << "), " \
        << "nrv: (" \
        << nrv[0] << ", " \
        << nrv[1] << "), " \
        << "(x, y): (" \
        << x << ", " \
        << y << "), " \
        << "prop: (" \
        << x_prop << ", " \
        << y_prop << ")\n" \
        << std::endl; \
        assert(test_); \
      }

  HM_ASSERT(tc[0] >= 0.0f);
  HM_ASSERT(tc[0] < old_width);
  HM_ASSERT(tc[1] >= 0.0f);
  HM_ASSERT(tc[1] <= old_height);

  //  FIXME - bitweeder
  //  Placeholder logic. We may need to determine the extrema of the
  //  rect that circumscribes the triangle and assign the appropriate coords
  //  based on where the input coords fall relative to those. Arguably, the
  //  current approach should be fine because we apply it consistently, but I
  //  suspect it presents a problem on chart boundaries since the charts may
  //  be rotated with respect to each other, rendering this simple tex coord
  //  approach useless.
  if (x_prop > 0.999 && y_prop > 0.999) {
    nrv = {x, y};
  } else if (x_prop > 0.999) {
    nrv = {x, y + texture_patch_border};
  } else if (y_prop > 0.999) {
    nrv = {x + texture_patch_border, y};
  } else {
    nrv = {x + texture_patch_border, y + texture_patch_border};
  }

  HM_ASSERT(nrv[0] >= texture_patch_border);
  HM_ASSERT(nrv[0] <= new_width - texture_patch_border);
  HM_ASSERT(nrv[1] >= texture_patch_border);
  HM_ASSERT(nrv[1] <= new_height - texture_patch_border);

  #undef HM_ASSERT

  return nrv;
}

/**
  @brief Moire-free image scaling.
 
  @details Smear the original texel over a grid up to 2x2 texels large based on
  the conversion from the original coordinates. Note that this algorithm is
  content-agnostic; every texel in the rescale area will be processed
  regardless of whether it contains any signal.
 
  SEEME - bitweeder
  This will play merry hell with texture coordinates, especially in an atlas,
  and especially if the neighboring charts are scaled at different rates or
  rotated relative to the adjacent adges. A more transformation-agnostic
  variant would use a 3x3 grid centered on the original texel.
*/
mve::FloatImage::Ptr rescale_area(
    mve::FloatImage::Ptr input_image,
    const int new_width,
    const int new_height);
mve::FloatImage::Ptr rescale_area(
    mve::FloatImage::Ptr input_image,
    const int new_width,
    const int new_height) {
  assert(!!input_image);
  assert(new_width <= input_image->width());
  assert(new_height <= input_image->height());
  
  auto out_image(mve::FloatImage::create());

  out_image->allocate(new_width, new_height, input_image->channels());
  out_image->fill(0.0);

  /*
    We outset the original extents to create a simulated border around the
    source image. This is done in order to improve the texel color result for
    border pixels in the chart when applying linear filtering, as otherwise
    they may end up too faint, in this case tending towards black. We account
    for the adjustment below.
  
//    Additionally, we inset the extents of the new rect so that there will be a
//    reserved border around the interior of the chart. This is to accommodate
//    the triangle masks, which assume they occupy a one-pixel border around a
//    given triangle.
  
    FIXME - bitweeder
    Note that the approach taken here is naive and branch-heavy. A better
    solution would incorporate tests transparently. This way is easy to
    reason about and test, and also benefits from expedience.
  */
  const auto old_width = input_image->width();
  const auto old_height = input_image->height();
  const auto v_old_width = input_image->width() + 2 * texture_patch_border;
  const auto v_old_height = input_image->height() + 2 * texture_patch_border;
  const auto v_new_width = new_width;// - 2 * texture_patch_border;
  const auto v_new_height = new_height;// - 2 * texture_patch_border;
  const float x_scale = v_new_width / static_cast<float>(v_old_width);
  const float y_scale = v_new_height / static_cast<float>(v_old_height);
  const float scale = x_scale * y_scale;

  for (int ci = 0; ci < input_image->channels(); ++ci) {
    for (int y = 0; y < v_old_height; ++y) {
      //  Keep track of where our actual y is.
      float y_low = static_cast<float>(y) * y_scale;// + texture_patch_border;
      int new_y = static_cast<int>(std::floor(y_low));

      //  Track where we are in the fractional pixel.
      float y_prop = std::min(1.0f, (std::floor(y_low) + 1.0f - y_low) / y_scale);

      for (int x = 0; x < v_old_width; ++x) {
        //  Keep track of where our actual x is.
        float x_low = static_cast<float>(x) * x_scale;// + texture_patch_border;
        int new_x = static_cast<int>(std::floor(x_low));

        //  Track where we are in the fractional pixel.
        float x_prop = std::min(1.0f, (std::floor(x_low) + 1.0f - x_low) / x_scale);

        float val = 0.0f;
        
        #define HM_ASSERT(test_) \
          if (!(test_)) {\
            std::cout << "failed: " << #test_ << std::endl; \
            std::cout \
              << "image dims: (" \
              << input_image->width() << ", " \
              << input_image->height() << "), " \
              << "new: (" \
              << new_width << ", " \
              << new_height << "), " \
              << "scale: (" \
              << x_scale << ", " \
              << y_scale << ")" \
              << "image coords: (" \
              << x << ", " << y << "), " \
              << "prop: (" \
              << x_prop << ", " << y_prop << "), " \
              << "val: " << val \
              << "max: (" << max_x << ", " << max_y << ")\n" \
              << std::endl; \
              assert(test_); \
          }

        //  This series of tests is intended to weed out our fake borders while
        //  still behaving as if they actually existed. The net impact is to
        //  ensure that border pixel colors are still properly normalized
        //  instead of being fragments.
        if (x < texture_patch_border) {
          if (y < texture_patch_border) {
            val = input_image->at(0, 0, ci) * scale;
          } else if (y >= v_old_height - texture_patch_border) {
            val = input_image->at(0, v_old_height - (2 * texture_patch_border + 1), ci) * scale;
          } else {
            val = input_image->at(0, y - texture_patch_border, ci) * scale;
          }
        } else if (x >= v_old_width - texture_patch_border) {
          if (y < texture_patch_border) {
            val = input_image->at(v_old_width - (2 * texture_patch_border + 1), 0, ci) * scale;
          } else if (y >= v_old_height - texture_patch_border) {
            val = input_image->at(v_old_width - (2 * texture_patch_border + 1), v_old_height - (2 * texture_patch_border + 1), ci) * scale;
          } else {
            val = input_image->at(v_old_width - (2 * texture_patch_border + 1), y - texture_patch_border, ci) * scale;
          }
        } else {
          if (y < texture_patch_border) {
            val = input_image->at(x - texture_patch_border, 0, ci) * scale;
          } else if (y >= v_old_height - texture_patch_border) {
            val = input_image->at(x - texture_patch_border, v_old_height - (2 * texture_patch_border + 1), ci) * scale;
          } else {
            val = input_image->at(x - texture_patch_border, y - texture_patch_border, ci) * scale;
          }
        }

        //  For this series of tests, we need to ensure that we only add
        //  `value` to texels that actually exist in our dest image. Note that
        //  this is basically just linear filtering routine with a fake border.
        if (x_prop > 0.999 && y_prop > 0.999) {
          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x) && (new_height > new_y)) {
            out_image->at(new_x, new_y, ci) += val;
          }
        } else if (x_prop > 0.999) {
          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x) && (new_height > new_y)) {
            out_image->at(new_x, new_y, ci) += val * y_prop;
          }
          
          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x) && (new_height > new_y + 1)) {
            out_image->at(new_x, new_y + 1, ci) += val * (1 - y_prop);
          }
        } else if (y_prop > 0.999) {
          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x) && (new_height > new_y)) {
            out_image->at(new_x, new_y, ci) += val * x_prop;
          }

          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x + 1) && (new_height > new_y)) {
            out_image->at(new_x + 1, new_y, ci) += val * (1 - x_prop);
          }
        } else {
          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x) && (new_height > new_y)) {
            out_image->at(new_x, new_y, ci) += val * x_prop * y_prop;
          }

          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x + 1) && (new_height > new_y)) {
            out_image->at(new_x + 1, new_y, ci) +=
                val * (1 - x_prop) * y_prop;
          }

          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x) && (new_height > new_y + 1)) {
            out_image->at(new_x, new_y + 1, ci) +=
                val * x_prop * (1 - y_prop);
          }
          
          if ((0 <= new_x) && (0 <= new_y)
              && (new_width > new_x + 1) && (new_height > new_y + 1)) {
            out_image->at(new_x + 1, new_y + 1, ci) +=
                val * (1 - x_prop) * (1 - y_prop);
          }
        }
        
        #undef HM_ASSERT
      }
    }
  }
  
  return out_image;
}

/**
  @brief Rescale a patch and underlying imagery.
 
  @details Note that the output image will be slightly larger than was actually
  requested due to the addition of a `texture_patch_border`-wide border around
  it, as per mvs-texturing’s expectation.
*/
void TexturePatch::rescale(double ratio) {
  int old_width = get_width();
  int old_height = get_height();
  int new_width = std::ceil(old_width * ratio) + 2 * texture_patch_border;
  int new_height = std::ceil(old_height * ratio) + 2 * texture_patch_border;

  //  SEEME - bitweeder
  //  It appears that there were moiré patterns being generated with the
  //  image scaling being done originally, necessitating an image scaling
  //  replacement function.
//  image = mve::image::rescale<float>(image,
//    mve::image::RescaleInterpolation::RESCALE_LINEAR, new_width, new_height);

  image = rescale_area(image, new_width, new_height);

  //  We recalculate the validity_mask and blending_mask from scratch to avoid
  //  rounding errors of all kinds.
  validity_mask = mve::ByteImage::create(get_width(), get_height(), 1);
  blending_mask = mve::ByteImage::create(get_width(), get_height(), 1);

  //  Strictly speaking, these calls end up being redundant.
  validity_mask->fill(0);
  blending_mask->fill(0);

  if (texcoords.size() >= 3) {
    //  We handle these as triples in case the relative positioning turns out
    //  to be important to avoid degeneration when applying scaling to the
    //  triangle.
    for (std::size_t i = 0; i < texcoords.size(); i += 3) {
      auto& v1 = texcoords[i];
      auto& v2 = texcoords[i + 1];
      auto& v3 = texcoords[i + 2];

      v1 = scale_texcoord(v1, old_width, old_height, new_width, new_height);
      v2 = scale_texcoord(v2, old_width, old_height, new_width, new_height);
      v3 = scale_texcoord(v3, old_width, old_height, new_width, new_height);
    }
  }
  
  std::vector<math::Vec3f> patch_adjust_values(
      get_faces().size() * 3, math::Vec3f(0.0f));

  adjust_colors(patch_adjust_values);
}

void TexturePatch::expose_blending_mask() {
  if (!!blending_mask) {
    for (int y = 0; y < image->height(); ++y) {
      for (int x = 0; x < image->width(); ++x) {
        image->at(x, y, 0) = image->at(x, y, 0) / 2.0f
            + static_cast<float>(blending_mask->at(x, y, 0)) / 255.0f / 2.0f;
        
        if ((0 == x) or (0 == y) or (image->width() - 1 == x) or (image->height() - 1 == y)) {
          image->at(x, y, 0) = image->at(x, y, 0) / 2.0f + 0.5;
          image->at(x, y, 1) = image->at(x, y, 1) / 2.0f + 0.5;
          image->at(x, y, 2) = image->at(x, y, 2) / 2.0f + 0.5;
        }
      }
    }
  }

  for (auto const& tc : texcoords) {
    image->at(tc[0], tc[1], 0) = 1.0;
    image->at(tc[0], tc[1], 1) = 1.0;
    image->at(tc[0], tc[1], 2) = 1.0;
  }
}

void TexturePatch::expose_validity_mask() {
  if (!!validity_mask) {
    for (int y = 0; y < image->height(); ++y) {
      for (int x = 0; x < image->width(); ++x) {
        image->at(x, y, 0) = image->at(x, y, 0) / 2.0f
            + static_cast<float>(validity_mask->at(x, y, 0)) / 255.0f / 2.0f;

        if ((0 == x) or (0 == y) or (image->width() - 1 == x) or (image->height() - 1 == y)) {
          image->at(x, y, 0) = image->at(x, y, 0) / 2.0f + 0.5;
          image->at(x, y, 1) = image->at(x, y, 1) / 2.0f + 0.5;
          image->at(x, y, 2) = image->at(x, y, 2) / 2.0f + 0.5;
        }
      }
    }
  }
  
  for (auto const& tc : texcoords) {
    image->at(tc[0], tc[1], 0) = 1.0;
    image->at(tc[0], tc[1], 1) = 1.0;
    image->at(tc[0], tc[1], 2) = 1.0;
  }
}

void TexturePatch::adjust_colors(
    std::vector<math::Vec3f> const& adjust_values,
    int num_channels) {
  regenerate_masks();

  if (texcoords.size() < 3) {
    return;
  }
  
  const float k_sqrt_2 = std::sqrt(2);

  mve::FloatImage::Ptr iadjust_values =
      mve::FloatImage::create(get_width(), get_height(), num_channels);

  for (std::size_t i = 0; i < texcoords.size(); i += 3) {
    auto const& v1 = texcoords[i];
    auto const& v2 = texcoords[i + 1];
    auto const& v3 = texcoords[i + 2];
    Tri tri {v1, v2, v3};
    auto area = tri.get_area();

    if (area < std::numeric_limits<float>::epsilon()) {
      continue;
    }

    auto aabb = tri.get_aabb();
    int const min_x = static_cast<int>(std::floor(aabb.min_x)) - texture_patch_border;
    int const min_y = static_cast<int>(std::floor(aabb.min_y)) - texture_patch_border;
    int const max_x = static_cast<int>(std::ceil(aabb.max_x)) + texture_patch_border;
    int const max_y = static_cast<int>(std::ceil(aabb.max_y)) + texture_patch_border;

    assert(0 <= min_x && max_x <= get_width());
    assert(0 <= min_y && max_y <= get_height());

    for (int y = min_y; y < max_y; ++y) {
      for (int x = min_x; x < max_x; ++x) {
        auto bcoords = tri.get_barycentric_coords(x, y);
        bool inside = bcoords.minimum() >= 0.0f;

        if (inside) {
          assert(x != 0 && y != 0);

          for (int c = 0; c < num_channels; ++c) {
            iadjust_values->at(x, y, c) = math::interpolate(
                adjust_values[i][c],
                adjust_values[i + 1][c],
                adjust_values[i + 2][c],
                bcoords[0],
                bcoords[1],
                bcoords[2]);
          }
        } else {
          if (validity_mask->at(x, y, 0) == 255) {
            continue;
          }

          //  Check whether the pixel’s distance from the triangle is more than
          //  one pixel.
          float ha = 2.0f * -bcoords[0] * area / (v2 - v3).norm();
          float hb = 2.0f * -bcoords[1] * area / (v1 - v3).norm();
          float hc = 2.0f * -bcoords[2] * area / (v1 - v2).norm();

          if (ha > k_sqrt_2 || hb > k_sqrt_2 || hc > k_sqrt_2) {
            continue;
          }
          
          for (int c = 0; c < num_channels; ++c) {
            iadjust_values->at(x, y, c) = math::interpolate(
                adjust_values[i][c],
                adjust_values[i + 1][c],
                adjust_values[i + 2][c],
                bcoords[0],
                bcoords[1],
                bcoords[2]);
          }
        }
      }
    }
  }

  if (num_channels <= 3) {
    for (int i = 0; i < image->get_pixel_amount(); ++i) {
      if (validity_mask->at(i, 0) != 0) {
        for (int c = 0; c < num_channels; ++c) {
          image->at(i, c) += iadjust_values->at(i, c);
        }
      } else {
        for (int c = 0; c < num_channels; ++c) {
          image->at(i, c) = 0.0f;
        }
      }
    }
  } else {
    for (int i = 0; i < image->get_pixel_amount(); ++i) {
      std::vector<float> raw_color(num_channels);

      if (validity_mask->at(i, 0) != 0) {
        std::copy(
            &image->at(i, 0),
            &image->at(i, 0) + num_channels,
            raw_color.begin());

        for (auto&& sub_color : raw_color) {
          sub_color += iadjust_values->at(i);
        }

        auto color = compute_object_class_color(&raw_color);

        std::copy(color.begin(), color.end(), &image->at(i, 0));
      } else {  // just set the rgb channels to 0
        math::Vec3f color {0.0f, 0.0f, 0.0f};

        std::copy(color.begin(), color.end(), &image->at(i, 0));
      }
    }
  }
}

void TexturePatch::regenerate_masks() {
  assert(!!blending_mask);
  assert(!!validity_mask);

  validity_mask->fill(0);

  if (texcoords.size() < 3) {
    return;
  }
  
  const float k_sqrt_2 = std::sqrt(2.0f);

  for (std::size_t i = 0; i < texcoords.size(); i += 3) {
    auto const& v1 = texcoords[i];
    auto const& v2 = texcoords[i + 1];
    auto const& v3 = texcoords[i + 2];
    Tri tri {v1, v2, v3};

    float area = tri.get_area();
    if (area < std::numeric_limits<float>::epsilon()) {
      continue;
    }

    auto aabb = tri.get_aabb();
    int const min_x = static_cast<int>(std::floor(aabb.min_x)) - texture_patch_border;
    int const min_y = static_cast<int>(std::floor(aabb.min_y)) - texture_patch_border;
    int const max_x = static_cast<int>(std::ceil(aabb.max_x)) + texture_patch_border;
    int const max_y = static_cast<int>(std::ceil(aabb.max_y)) + texture_patch_border;

    assert(0 <= min_x && max_x <= get_width());
    assert(0 <= min_y && max_y <= get_height());

    for (int y = min_y; y < max_y; ++y) {
      for (int x = min_x; x < max_x; ++x) {
        math::Vec3f bcoords = tri.get_barycentric_coords(x, y);
        bool inside = bcoords.minimum() >= 0.0f;
        if (inside) {
          assert(x != 0 && y != 0);

          validity_mask->at(x, y, 0) = 255;
          blending_mask->at(x, y, 0) = 255;
        } else {
          if (validity_mask->at(x, y, 0) == 255) {
            continue;
          }

          //  Check whether the pixel’s distance from the triangle is more than
          //  one pixel.
          float ha = 2.0f * -bcoords[0] * area / (v2 - v3).norm();
          float hb = 2.0f * -bcoords[1] * area / (v1 - v3).norm();
          float hc = 2.0f * -bcoords[2] * area / (v1 - v2).norm();

          if (ha > k_sqrt_2 || hb > k_sqrt_2 || hc > k_sqrt_2) {
            continue;
          }
          
          validity_mask->at(x, y, 0) = 255;
          blending_mask->at(x, y, 0) = 64;
        }
      }
    }
  }
}

bool TexturePatch::valid_pixel(math::Vec2f pixel) const {
  float x = pixel[0];
  float y = pixel[1];

  auto const height = static_cast<float>(get_height());
  auto const width = static_cast<float>(get_width());

  bool valid = (0.0f <= x && x < width && 0.0f <= y && y < height);
  if (valid && validity_mask != nullptr) {
    /* Only pixel which can be correctly interpolated are valid. */
    float cx = std::max(0.0f, std::min(width - 1.0f, x));
    float cy = std::max(0.0f, std::min(height - 1.0f, y));
    int const floor_x = static_cast<int>(cx);
    int const floor_y = static_cast<int>(cy);
    int const floor_xp1 = std::min(floor_x + 1, get_width() - 1);
    int const floor_yp1 = std::min(floor_y + 1, get_height() - 1);

    float const w1 = cx - static_cast<float>(floor_x);
    float const w0 = 1.0f - w1;
    float const w3 = cy - static_cast<float>(floor_y);
    float const w2 = 1.0f - w3;

    valid =
        (w0 * w2 == 0.0f || validity_mask->at(floor_x, floor_y, 0) == 255)
        && (w1 * w2 == 0.0f || validity_mask->at(floor_xp1, floor_y, 0) == 255)
        && (w0 * w3 == 0.0f || validity_mask->at(floor_x, floor_yp1, 0) == 255)
        && (w1 * w3 == 0.0f
            || validity_mask->at(floor_xp1, floor_yp1, 0) == 255);
  }

  return valid;
}

bool TexturePatch::valid_pixel(math::Vec2i pixel) const {
  int const x = pixel[0];
  int const y = pixel[1];

  bool valid = (0 <= x && x < get_width() && 0 <= y && y < get_height());
  if (valid && validity_mask != nullptr) {
    valid = validity_mask->at(x, y, 0) == 255;
  }

  return valid;
}

math::Vec3f TexturePatch::get_pixel_value(math::Vec2f pixel) const {
  assert(valid_pixel(pixel));

  math::Vec3f color;
  image->linear_at(pixel[0], pixel[1], *color);
  return color;
}

std::vector<float> TexturePatch::get_pixel_value_n(
    math::Vec2f pixel,
    int num_channels) const {
  // TODO dwh: for some reason this fails here but not above
  //    assert(valid_pixel(pixel));

  std::vector<float> color(num_channels);
  for (int i = 0; i < num_channels; i++) {
    color[i] = image->linear_at(pixel[0], pixel[1], i);
  }
  return color;
}

double TexturePatch::compute_geometric_area(
    const std::vector<math::Vec3f>& vertices,
    const std::vector<uint>& mesh_faces) const {
  double total_area = 0;
  for (auto f : faces) {
    total_area += std::abs(math::geom::triangle_area(
        vertices[mesh_faces[f * 3 + 0]],
        vertices[mesh_faces[f * 3 + 1]],
        vertices[mesh_faces[f * 3 + 2]]));
  }
  return total_area;
}

double TexturePatch::compute_pixel_area() const {
  double total_area = 0;
  for (std::size_t i = 0; i < faces.size(); ++i) {
    Tri tri(texcoords[i * 3], texcoords[i * 3 + 1], texcoords[i * 3 + 2]);
    total_area += tri.get_area();
  }
  return total_area;
}

void TexturePatch::set_pixel_value(math::Vec2i pixel, math::Vec3f color) {
  assert(blending_mask != NULL);
  assert(valid_pixel(pixel));

  std::copy(color.begin(), color.end(), &image->at(pixel[0], pixel[1], 0));
  blending_mask->at(pixel[0], pixel[1], 0) = 128;
}

void TexturePatch::set_pixel_value(
    math::Vec2i pixel,
    const std::vector<float>* all_channels) {
  assert(blending_mask != NULL);
  assert(valid_pixel(pixel));
  // Only copy the color channels
  // TODO dwh: remove hard-coded number of colors=3
  auto num_colors = std::min(static_cast<int>(all_channels->size()), 3);
  std::copy(
      all_channels->begin(),
      all_channels->begin() + num_colors,
      &image->at(pixel[0], pixel[1], 0));
  blending_mask->at(pixel[0], pixel[1], 0) = 128;
}

// TODO dwh: pass in an object to color mapping structure
math::Vec3f TexturePatch::compute_object_class_color(
    const std::vector<float>* color) {
  // TODO dwh: remove hard-coded number of colors=3
  auto num_colors = std::min(static_cast<int>(color->size()), 3);
  long arg_max = std::distance(
      color->begin() + num_colors,
      std::max_element(color->begin() + num_colors, color->end()));
  math::Vec3f final_class_color(0, 0, 0);
  // TODO !!! map colors from passed argument to method
  // TODO scale by value?
  //    REDUCE_MAP = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (205,
  //    133, 63), 4: (255, 255, 0), 5: (255, 255, 255), 6: (0, 0, 255)}
  switch (arg_max) {
    case 0: {
      math::Vec3f class_color(0.f, 0.f, 0.f);
      final_class_color = class_color;
      break;
    }
    case 1: {
      math::Vec3f class_color(1.f, 0.f, 0.f);
      final_class_color = class_color;
      break;
    }
    case 2: {
      math::Vec3f class_color(0.f, 1.f, 0.f);
      final_class_color = class_color;
      break;
    }
    case 3: {  // TODO divide these out???
      math::Vec3f class_color(205.f / 255.f, 133.f / 255.f, 63.f / 255.f);
      final_class_color = class_color;
      break;
    }
    case 4: {
      math::Vec3f class_color(1.f, 1.f, 0.f);
      final_class_color = class_color;
      break;
    }
    case 5: {
      math::Vec3f class_color(1.f, 1.f, 1.f);
      final_class_color = class_color;
      break;
    }
    case 6: {
      math::Vec3f class_color(0.f, 0.f, 1.f);
      final_class_color = class_color;
      break;
    }
    default: {
      math::Vec3f class_color(0.f, 0.f, 0.f);
      final_class_color = class_color;
      std::cout << "ERROR!! Bad class from " << color << " is " << arg_max
                << std::endl;
    }
  }
  return final_class_color;  // * color[arg_max];
}

void TexturePatch::set_pixel_object_class_value(
    math::Vec2i pixel,
    const std::vector<float>* color) {
  assert(valid_pixel(pixel));
  math::Vec3f final_class_color =
      TexturePatch::compute_object_class_color(color);
  //    std::cout << "Class from " << color << " is " << arg_max << " and class
  //    color is " << final_class_color << std::endl;
  set_pixel_value(pixel, final_class_color);
}

void TexturePatch::blend(mve::FloatImage::ConstPtr orig) {
  poisson_blend(orig, blending_mask, image, 1.0f);

  /* Invalidate all pixels outside the boundary. */
  for (int y = 0; y < blending_mask->height(); ++y) {
    for (int x = 0; x < blending_mask->width(); ++x) {
      if (blending_mask->at(x, y, 0) == 64) {
        validity_mask->at(x, y, 0) = 0;
      }
    }
  }
}

typedef std::vector<std::pair<int, int>> PixelVector;
typedef std::set<std::pair<int, int>> PixelSet;

void TexturePatch::prepare_blending_mask(std::size_t strip_width) {
  int const width = blending_mask->width();
  int const height = blending_mask->height();

  /* Calculate the set of valid pixels at the border of texture patch. */
  PixelSet valid_border_pixels;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (validity_mask->at(x, y, 0) == 0)
        continue;

      /* Valid border pixels need no invalid neighbours. */
      if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        valid_border_pixels.insert(std::pair<int, int>(x, y));
        continue;
      }

      /* Check the direct neighbourhood of all invalid pixels. */
      for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
          int nx = x + i;
          int ny = y + j;
          /* If the valid pixel has a invalid neighbour: */
          if (validity_mask->at(nx, ny, 0) == 0) {
            /* Add the pixel to the set of valid border pixels. */
            valid_border_pixels.insert(std::pair<int, int>(x, y));
          }
        }
      }
    }
  }

  mve::ByteImage::Ptr inner_pixel = validity_mask->duplicate();

  /* Iteratively erode all border pixels. */
  for (std::size_t i = 0; i < strip_width; ++i) {
    PixelVector new_invalid_pixels(
        valid_border_pixels.begin(), valid_border_pixels.end());
    PixelVector::iterator it;
    valid_border_pixels.clear();

    /* Mark the new invalid pixels invalid in the validity mask. */
    for (it = new_invalid_pixels.begin(); it != new_invalid_pixels.end();
         ++it) {
      int x = it->first;
      int y = it->second;

      inner_pixel->at(x, y, 0) = 0;
    }

    /* Calculate the set of valid pixels at the border of the valid area. */
    for (it = new_invalid_pixels.begin(); it != new_invalid_pixels.end();
         ++it) {
      int x = it->first;
      int y = it->second;

      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          int nx = x + i;
          int ny = y + j;
          if (0 <= nx && nx < width && 0 <= ny && ny < height
              && inner_pixel->at(nx, ny, 0) == 255) {
            valid_border_pixels.insert(std::pair<int, int>(nx, ny));
          }
        }
      }
    }
  }

  /* Sanitize blending mask. */
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      if (blending_mask->at(x, y, 0) == 128) {
        uint8_t n[] = {blending_mask->at(x - 1, y, 0),
                       blending_mask->at(x + 1, y, 0),
                       blending_mask->at(x, y - 1, 0),
                       blending_mask->at(x, y + 1, 0)};
        bool valid = true;
        for (uint8_t v : n) {
          if (v == 255)
            continue;
          valid = false;
        }
        if (valid)
          blending_mask->at(x, y, 0) = 255;
      }
    }
  }

  /* Mark all remaining pixels invalid in the blending_mask. */
  for (int i = 0; i < inner_pixel->get_pixel_amount(); ++i) {
    if (inner_pixel->at(i) == 255)
      blending_mask->at(i) = 0;
  }

  /* Mark all border pixels. */
  PixelSet::iterator it;
  for (it = valid_border_pixels.begin(); it != valid_border_pixels.end();
       ++it) {
    int x = it->first;
    int y = it->second;

    blending_mask->at(x, y, 0) = 128;
  }
}
