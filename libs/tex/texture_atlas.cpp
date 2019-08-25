/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <map>
#include <set>

#include <mve/image_io.h>
#include <mve/image_tools.h>
#include <util/file_system.h>

#include "settings.h"
#include "texture_atlas.h"

TextureAtlas::TextureAtlas(unsigned int size)
  : size {size} {
  bin = RectangularBin::create(size, size);
  image = mve::ByteImage::create(size, size, 3);
  validity_mask = mve::ByteImage::create(size, size, 1);
}

/**
 * Copies the src image into the dest image at the given position,
 * optionally adding a border.
 * @warning asserts that the given src image fits into the given dest image.
 */
void copy_into(
    mve::ByteImage::ConstPtr src,
    int x,
    int y,
    mve::ByteImage::Ptr dest,
    int border = 0) {
  assert(x >= 0 && x + src->width() + 2 * border <= dest->width());
  assert(y >= 0 && y + src->height() + 2 * border <= dest->height());

  for (int i = 0; i < src->width() + 2 * border; ++i) {
    for (int j = 0; j < src->height() + 2 * border; j++) {
      int sx = i - border;
      int sy = j - border;

      if (sx < 0 || sx >= src->width() || sy < 0 || sy >= src->height())
        continue;

      for (int c = 0; c < src->channels(); ++c) {
        dest->at(x + i, y + j, c) = src->at(sx, sy, c);
      }
    }
  }
}

typedef std::vector<std::pair<int, int>> PixelVector;
typedef std::set<std::pair<int, int>> PixelSet;

uint TextureAtlas::insert(TexturePatch::Ptr texture_patch) {
  if (finalized) {
    throw util::Exception(
        "No insertion possible, TextureAtlas already finalized");
  }

  assert(bin != NULL);
  assert(validity_mask != NULL);

  uint local_padding = compute_local_padding(
      texture_patch->get_width(), texture_patch->get_height(), size);

  uint const width = texture_patch->get_width() + 2 * local_padding;
  uint const height = texture_patch->get_height() + 2 * local_padding;

//  std::cout << "padding: " << compute_base_padding(size)
//      << ", local_padding: " << local_padding
//      << ", width: " << width
//      << ", tpwidth(): " << texture_patch->get_width()
//      << ", height: " << height
//      << ", tpheight(): " << texture_patch->get_height()
//      << ", expected area: " << width * height
//      << std::flush;

  Rect<int> rect(0, 0, width, height);
  auto area = bin->insert(&rect) ? width * height : 0;

//  std::cout << ", actual area: " << area << std::endl;

  if (0 == area) return area;

  /* Update texture atlas and its validity mask. */
  mve::ByteImage::Ptr patch_image =
      mve::image::float_to_byte_image(texture_patch->get_image(), 0.0f, 1.0f);

  copy_into(
      patch_image,
      rect.min_x,
      rect.min_y,
      image,
      local_padding);

  mve::ByteImage::ConstPtr patch_validity_mask =
      texture_patch->get_validity_mask();

  copy_into(
      patch_validity_mask,
      rect.min_x,
      rect.min_y,
      validity_mask,
      local_padding);

  TexturePatch::Faces const& patch_faces = texture_patch->get_faces();
  TexturePatch::Texcoords const& patch_texcoords =
      texture_patch->get_texcoords();

  /* Calculate the offset of the texture patches' relative texture coordinates
   */
  math::Vec2f offset =
      math::Vec2f(rect.min_x + local_padding, rect.min_y + local_padding);

  faces.insert(faces.end(), patch_faces.begin(), patch_faces.end());

  /* Calculate the final textcoords of the faces. */
  for (std::size_t i = 0; i < patch_faces.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      math::Vec2f rel_texcoord(patch_texcoords[i * 3 + j]);
      math::Vec2f texcoord = rel_texcoord + offset;

      texcoord[0] = texcoord[0] / this->size;
      texcoord[1] = texcoord[1] / this->size;
      texcoords.push_back(texcoord);
    }
  }
  
  return area;
}

/**
  @brief Generate padding pixels bordering each chart on the atlas page.
  @details Effectively, we radially dilate the values of every border pixel of
  every texture chart in a breadth-first manner until we encounter another
  padding pixel, a valid pixel, or the texture edge. Dilation will be performed
  out to a maximum distance of the base padding distance; this will be at least
  as large as the local padding set aside around each chart, but given the
  aforementioned termination conditions will not cause a conflict.
*/
void TextureAtlas::apply_edge_padding(tex::Settings const& settings) {
  assert(!!image);
  assert(!!validity_mask);

  if (settings.dilate_padding_pixels) {
    std::cout << "Applying edge padding" << std::endl;
  } else {
    std::cout << "Not applying edge padding" << std::endl;
    return;
  }

  const int width = image->width();
  const int height = image->height();

  assert(width == validity_mask->width());
  assert(height == validity_mask->height());

  math::Matrix<float, 3, 3> gauss {};

  gauss[0] = 1.0f;
  gauss[1] = 2.0f;
  gauss[2] = 1.0f;
  gauss[3] = 2.0f;
  gauss[4] = 4.0f;
  gauss[5] = 2.0f;
  gauss[6] = 1.0f;
  gauss[7] = 2.0f;
  gauss[8] = 1.0f;
  
  gauss /= 16.0f;

  //  Calculate the set of invalid pixels at the border of texture patches.
  PixelSet invalid_border_pixels {};
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (validity_mask->at(x, y, 0) == 255)
        continue;

      /* Check the direct neighbourhood of all invalid pixels. */
      for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
          int nx = x + i;
          int ny = y + j;
          /* If the invalid pixel has a valid neighbour: */
          if (0 <= nx && nx < width && 0 <= ny && ny < height
              && validity_mask->at(nx, ny, 0) == 255) {
            /* Add the pixel to the set of invalid border pixels. */
            invalid_border_pixels.insert(std::pair<int, int>(x, y));
          }
        }
      }
    }
  }
  
  mve::ByteImage::Ptr new_validity_mask = validity_mask->duplicate();

  /* Iteratively dilate border pixels until padding constants are reached. */
  auto padding = compute_base_padding(size);

  for (unsigned int n = 0; n <= padding; ++n) {
    PixelVector new_valid_pixels {};
    auto it = invalid_border_pixels.begin();

    for (; it != invalid_border_pixels.end(); it++) {
      int x = it->first;
      int y = it->second;
      bool now_valid = false;

      /* Calculate new pixel value. */
      for (int c = 0; c < 3; ++c) {
        float norm = 0.0f;
        float value = 0.0f;
        
        for (int j = -1; j <= 1; ++j) {
          for (int i = -1; i <= 1; ++i) {
            int nx = x + i;
            int ny = y + j;
            
            if (0 <= nx && nx < width && 0 <= ny && ny < height
                && new_validity_mask->at(nx, ny, 0) == 255) {
              float w = gauss[(j + 1) * 3 + (i + 1)];
              norm += w;
              value += (image->at(nx, ny, c) / 255.0f) * w;
            }
          }
        }

        if (norm == 0.0f) {
          continue;
        }

        now_valid = true;
        
        if (settings.highlight_padding_pixels) {
          image->at(x, y, c) = (c == 1) ? 255 : 0;
        } else {
          image->at(x, y, c) = (value / norm) * 255.0f;
        }
      }

      if (now_valid) {
        new_valid_pixels.push_back(*it);
      }
    }

    invalid_border_pixels.clear();

    /* Mark the new valid pixels valid in the validity mask. */
    for (std::size_t i = 0; i < new_valid_pixels.size(); ++i) {
      int x = new_valid_pixels[i].first;
      int y = new_valid_pixels[i].second;

      new_validity_mask->at(x, y, 0) = 255;
    }

    //  Calculate the set of invalid pixels at the border of the valid area.
    for (std::size_t i = 0; i < new_valid_pixels.size(); ++i) {
      int x = new_valid_pixels[i].first;
      int y = new_valid_pixels[i].second;

      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          int nx = x + k;
          int ny = y + j;
          
          if (0 <= nx && nx < width && 0 <= ny && ny < height
              && new_validity_mask->at(nx, ny, 0) == 0) {
            invalid_border_pixels.insert(std::pair<int, int>(nx, ny));
          }
        }
      }
    }
  }
}

struct VectorCompare {
  bool operator()(math::Vec2f const& lhs, math::Vec2f const& rhs) const {
    return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
  }
};

typedef std::map<math::Vec2f, std::size_t, VectorCompare> TexcoordMap;

void TextureAtlas::merge_texcoords() {
  Texcoords tmp;
  tmp.swap(this->texcoords);

  TexcoordMap texcoord_map;
  for (math::Vec2f const& texcoord : tmp) {
    TexcoordMap::iterator iter = texcoord_map.find(texcoord);
    if (iter == texcoord_map.end()) {
      std::size_t texcoord_id = this->texcoords.size();
      texcoord_map[texcoord] = texcoord_id;
      this->texcoords.push_back(texcoord);
      this->texcoord_ids.push_back(texcoord_id);
    } else {
      this->texcoord_ids.push_back(iter->second);
    }
  }
}

void TextureAtlas::finalize(tex::Settings const& settings) {
  if (finalized) {
    throw util::Exception("TextureAtlas already finalized");
  }

  this->bin.reset();
  this->apply_edge_padding(settings);
  this->validity_mask.reset();
  this->merge_texcoords();

  this->finalized = true;
}
