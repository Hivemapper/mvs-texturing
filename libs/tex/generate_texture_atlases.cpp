/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */


#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include <mve/image_tools.h>
#include <util/timer.h>

#include "defines.h"
#include "histogram.h"
#include "settings.h"
#include "texturing.h"
#include "texture_atlas.h"
#include "texture_patch.h"
#include "texturing.h"

//  FIXME - bitweeder
//  It’s unclear what the significance of these magic numbers is supposed to
//  be, e.g., “maximum dimension size allowed for a POT texture, as imposed by
//  the graphics API”. In context, the bounding rect of a given texture chart
//  plus its padding is not allowed to exceed MAX_TEXTURE_SIZE in either the
//  x or y dimension, but it’s unclear where this limitation comes from.
//  PREF_TEXTURE_SIZE is even more mysterious.
#define MAX_TEXTURE_SIZE (16 * 1024)
#define PREF_TEXTURE_SIZE (16 * 1024)
#define MIN_TEXTURE_SIZE (256)

TEX_NAMESPACE_BEGIN

/**
  @brief Pull magic numbers out of Moehrle’s ass
  @details Heuristic to calculate an appropriate texture atlas size.
  @warning asserts that no texture patch exceeds the dimensions
  of the maximal possible texture atlas size.
 
  @param texture_patches is a vector of charts to place in the atlas page
 
  @return the edge length of the texture square, in texels

  FIXME - bitweeder
  There is way too much inexplicable stuff going on here.
*/
struct AtlasPageEsts {
  uint edge_length {};
  uint occupied_area {};
  uint max_chart_width{};
  uint max_chart_height{};
};

AtlasPageEsts compute_page_estimates(ConstTexturePatches const& texture_patches);
AtlasPageEsts compute_page_estimates(ConstTexturePatches const& texture_patches) {
  AtlasPageEsts nrv = {MAX_TEXTURE_SIZE, 0, 0, 0};

  while (true) {
    nrv.occupied_area = 0;
    nrv.max_chart_width = 0;
    nrv.max_chart_height = 0;

    //  FIXME - bitweeder
    //  It’s a kind of magic.
    //  The padding size is dependent upon the edge length, so each iteration
    //  through the loop - which more tightly bounds edge_length - will also
    //  (potentially) generate different values for our dependent variables.
    //  There is no explanation for how this padding calculation was derived.
    //  It’s worth noting that even with anisotropic filtering, it’s unlikely
    //  we’d ever need more than a 2-pixel border unless something has gone
    //  horribly wrong, so this aspect is even more confusing.
    for (auto const& texture_patch : texture_patches) {
      auto const tpwidth = texture_patch->get_width();
      auto const tpheight = texture_patch->get_height();
      auto local_padding = compute_local_padding(tpwidth, tpheight, nrv.edge_length);
      uint width = tpwidth + 2 * local_padding;
      uint height = tpheight + 2 * local_padding;
      uint area = width * height;

      nrv.occupied_area += area;
      nrv.max_chart_width = std::max(nrv.max_chart_width, width);
      nrv.max_chart_height = std::max(nrv.max_chart_height, height);
    }

    //  FIXME - bitweeder
    //  Asserts seem a bit over the top here since, as written, it was not
    //  previously possible to know whether we had a legal chart or not. This
    //  should be rewritten to be more robust, e.g., by making the max
    //  dimensions known to callers.
    assert(nrv.max_chart_width <= MAX_TEXTURE_SIZE);
    assert(nrv.max_chart_height <= MAX_TEXTURE_SIZE);

    //  FIXME - bitweeder
    //  WTF is up with that last criterion?
    if (nrv.edge_length > PREF_TEXTURE_SIZE
     && nrv.max_chart_width < PREF_TEXTURE_SIZE
     && nrv.max_chart_height < PREF_TEXTURE_SIZE
     && nrv.occupied_area / (PREF_TEXTURE_SIZE * PREF_TEXTURE_SIZE) < 8) {
      nrv.edge_length = PREF_TEXTURE_SIZE;
      continue;
    }

    if (nrv.edge_length <= MIN_TEXTURE_SIZE) {
      nrv.edge_length = MIN_TEXTURE_SIZE;
      break;
    }

    //  FIXME - bitweeder
    //  Seriosuly, WTF?
    if (nrv.max_chart_height < nrv.edge_length / 2
     && nrv.max_chart_width < nrv.edge_length / 2
     && static_cast<double>(nrv.occupied_area) /
        static_cast<double>(nrv.edge_length * nrv.edge_length) < 0.2) {
      nrv.edge_length = nrv.edge_length / 2;
      continue;
    }
    
    break;
  }

  return nrv;
}

void generate_capped_texture_atlas(
    TexturePatches* orig_texture_patches,
    Settings const& settings,
    TextureAtlases* texture_atlases,
    uint max_atlas_dim,
    const std::vector<math::Vec3f>& vertices,
    const std::vector<uint>& faces) {
  ConstTexturePatches texture_patches {};

  //  FIXME - bitweeder
  //  This assumes a specific gamma correction which may be inappropriate for
  //  the source images.
  std::transform(orig_texture_patches->begin(), orig_texture_patches->end(),
      texture_patches.begin(), [tm=settings.tone_mapping](auto&& patch) {
        if (tm != TONE_MAPPING_NONE) {
          mve::image::gamma_correct(patch->get_image(), 1.0f / 2.2f);
        }
        
        return patch;
      });

  orig_texture_patches->clear();

  //  Improve the bin-packing algorithm efficiency by sorting texture patches
  //  in descending order of size.
  //
  //  SEEME - bitweeder
  //  This depends upon the size of the bounding rect, as opposed to the actual
  //  patch area. Furthermore, it’s a straight area sort, not even a longest
  //  dimension sort, much less something fancier.
  std::sort(texture_patches.begin(), texture_patches.end(),
      [](auto const& lhs, auto const& rhs) {
        return lhs->get_size() > rhs->get_size();
      });

  //  We start with full-size rendering, and scale downwards if the charts
  //  won’t fit on a single atlas page. The amount we scale depends on how
  //  much area was successfully transferred to the page in the previous
  //  iteration relative to how much we expected to transfer; the page is
  //  effectively cleared between iterations.
  //
  //  SEEME - bitweeder
  //  We use an ad hoc metric for determining scaling, intending to converge on
  //  a useful value. Note that it is not feasible to be precise, as
  //  complications ensue depending on the relative number of small charts,
  //  the size of the chart borders, the packing algorithm being used, and
  //  possibly the shape of the charts.
  double scaling = 1.0;
  std::size_t iterations = 0;

  while (true) {
    //  estimated_size is one dimension of the bounding square of the collected
    //  atlas charts after packing (in other words, square textures are
    //  assumed). Note that estimated_size is just a starting point; for
    //  various reasons, it may be completely off.
    auto atlas_page_ests = compute_page_estimates(texture_patches);
    auto atlas_size = (std::min(atlas_page_ests.edge_length, max_atlas_dim));
    auto texture_atlas = TextureAtlas::create(atlas_size);
    bool atlas_complete = true;
    uint actual_occupied_area = 0;

    for (std::size_t i = 0; i < texture_patches.size(); ++i) {
      ++iterations;
      
      uint occupied_area = 0;
      
      if (scaling == 1.0) {
        occupied_area = texture_atlas->insert(texture_patches[i]);
      } else {
        //  Generate a deep copy of the original patch.
        auto patch = TexturePatch::create(texture_patches[i]);

        patch->rescale(scaling);

        //  Attempt to insert the patch.
        occupied_area = texture_atlas->insert(patch);
      }

      if (0 == occupied_area) {
        atlas_complete = false;
        break;
      } else {
        actual_occupied_area += occupied_area;
      }
    }

    if (atlas_complete) {
      //  FIXME - bitweeder
      //  Prettify this floating point value.
      std::cout << "Completed atlas page with "
                << scaling * 100.0 << "% scaling" << std::endl;

      texture_atlas->finalize();
      texture_atlases->push_back(texture_atlas);
      break;
    } else {
      //  FIXME - bitweeder
      //  Prettify this floating point value.
      std::cout << "Unable to complete atlas page with "
                << scaling * 100.0 << "% scaling. Adjusting." << std::endl;

      scaling *= (static_cast<double>(actual_occupied_area)
          / static_cast<double>(atlas_page_ests.occupied_area));
    }
    
    if ((scaling < 0.01) || (iterations >= 10)) {
      std::cout << "Unable to complete atlas page at all" << std::endl;
      return;
    }
  }
}

void generate_texture_atlases(
    TexturePatches* orig_texture_patches,
    Settings const& settings,
    TextureAtlases* texture_atlases,
    const std::vector<math::Vec3f>& vertices,
    const std::vector<uint>& faces) {
  ConstTexturePatches texture_patches {};

  //  FIXME - bitweeder
  //  This assumes a specific gamma correction which may be inappropriate for
  //  the source images.
  std::transform(orig_texture_patches->begin(), orig_texture_patches->end(),
      texture_patches.begin(), [tm=settings.tone_mapping](auto&& patch) {
        if (tm != TONE_MAPPING_NONE) {
          mve::image::gamma_correct(patch->get_image(), 1.0f / 2.2f);
        }
        
        return patch;
      });

  orig_texture_patches->clear();

  //  Improve the bin-packing algorithm efficiency by sorting texture patches
  //  in descending order of size.
  //
  //  SEEME - bitweeder
  //  This depends upon the size of the bounding rect, as opposed to the actual
  //  patch area.
  std::sort(texture_patches.begin(), texture_patches.end(),
      [](auto const& lhs, auto const& rhs) {
        return lhs->get_size() > rhs->get_size();
      });

  auto const total_num_patches = texture_patches.size();
  auto remaining_patches = texture_patches.size();

  std::ofstream tty("/dev/tty", std::ios_base::out);

  #pragma omp parallel
  {
    #pragma omp single
    {
      while (!texture_patches.empty()) {
        auto atlas_page_ests = compute_page_estimates(texture_patches);

        texture_atlases->push_back(TextureAtlas::create(atlas_page_ests.edge_length));

        auto texture_atlas = texture_atlases->back();

        //  Try to insert each of the texture patches into the texture atlas.
        for (auto it = texture_patches.begin(); it != texture_patches.end();) {
          std::size_t done_patches = total_num_patches - remaining_patches;
          int percent =
              static_cast<float>(done_patches) / total_num_patches * 100.0f;

          if (total_num_patches > 100
              && done_patches % (total_num_patches / 100) == 0) {
            tty << "\r\tWorking on atlas " << texture_atlases->size() << " "
                << percent << "%... " << std::flush;
          }

          if (texture_atlas->insert(*it)) {
            it = texture_patches.erase(it);
            remaining_patches -= 1;
          } else {
            ++it;
          }
        }

        #pragma omp task
        texture_atlas->finalize();
      }

      std::cout << "\r\tWorking on atlas " << texture_atlases->size()
                << " 100%... done." << std::endl;

      util::WallTimer timer {};
      std::cout << "\tFinalizing texture atlases... " << std::flush;

      #pragma omp taskwait
      std::cout << "done. (Took: " << timer.get_elapsed_sec() << "s)"
                << std::endl;

      /* End of single region */
    }
    /* End of parallel region. */
  }
}

TEX_NAMESPACE_END
