#include "mask/euclidean_view_mask.h"

#include <iostream>

#include "mask/triangle_voxel_intersection.h"

using std::vector;
using std::set;

namespace MvsTexturing {

EuclideanViewMask::EuclideanViewMask() {
  vmin = Eigen::Matrix<double, 3, 1>::Zero();
  coord_transform = Eigen::Matrix<double, 3, 3>::Identity();
}

EuclideanViewMask::EuclideanViewMask(const Eigen::Matrix<double, 3, 1>& vmin,
                                     const Eigen::Matrix<double, 3, 3>& coord_transform,
                                     int nx,
                                     int ny)
: vmin(vmin),
  coord_transform(coord_transform) {

  mask_data.resize(nx);
  for (int i = 0; i < nx; ++i) {
    mask_data[i].resize(ny);
  }
}

/**
 * @brief Returns true of x,y is within the nx * ny grid
 */
bool EuclideanViewMask::isValidXy(int x, int y) const {
  return (x >= 0 && x < mask_data.size() && y >= 0 && y < mask_data[0].size());
}

/**
 * @brief Takes a 3d vector coordinate, converts it to the local coordinate system and returns
 * the local voxel index
 */
vector<int> EuclideanViewMask::getVoxelIndex(const Eigen::Matrix<double, 3, 1>& v) const {
  Eigen::Matrix<double, 3, 1> vi = coord_transform*(v - vmin);
  vector<int> xyz(3);
  xyz[0] = floor(vi[0]);
  xyz[1] = floor(vi[1]);
  xyz[2] = floor(vi[2]);
  if (!isValidXy(xyz[0], xyz[1])) {
    std::cout << "Warning: location " << v.transpose() << " results in invalid coordinates " << std::endl;
    throw;
  }
  return xyz;
}

/**
 * @brief (private) non const accessor/mutator
 */
set<uint16_t>& EuclideanViewMask::get(const vector<int>& xyz) {
  if (!isValidXy(xyz[0], xyz[1])) {
    std::cout << "invalid coordinates " << std::endl;
    throw;
  }
  return mask_data[xyz[0]][xyz[1]][xyz[2]];
}

/**
 * @brief const accessor
 */
const set<uint16_t>& EuclideanViewMask::operator[](const vector<int>& xyz) const {
  if (isValidXy(xyz[0], xyz[1]))
    return mask_data[xyz[0]][xyz[1]].at(xyz[2]);
  else
    throw;
}

void EuclideanViewMask::append(const Eigen::Matrix<double, 3, 1>& v, uint16_t i) {
  set<uint16_t>& voxel = get(getVoxelIndex(v));
  voxel.insert(i);
}

void EuclideanViewMask::append(const Eigen::Matrix<double, 3, 1>& v,  const set<uint16_t>& is) {
  set<uint16_t>& voxel = get(getVoxelIndex(v));
  for (uint16_t i : is) {
    voxel.insert(i);
  }
}

/**
 * @brief Finds all voxels intersecting the triangle given by vertices
 * 
 * @param voxels Output list of voxel indices
 */
void EuclideanViewMask::getTriangleVoxels(const vector<Eigen::Matrix<double, 3, 1>>& vertices,
                                          vector<vector<int>>& voxels) const {
  assert(vertices.size() == 3);
  vector<int> mins(3, INFINITY);
  vector<int> maxes(3, -INFINITY);
  // compute bounding box
  for (int i = 0; i < 3; ++i) {
    vector<int> vv = getVoxelIndex(vertices[i]);
    for (int j = 0; j < 3; ++j) {
      if (vv[j] < mins[j])
        mins[j] = vv[j];
      if (vv[j] > maxes[j])
        maxes[j] = vv[j];
    }
  }
  for (int x = mins[0]; x <= maxes[0]; ++x) {
    for (int y = mins[1]; y <= maxes[1]; ++y) {
      for (int z = mins[2]; z <= maxes[2]; ++z) {
        TriangleCell::Triangle3 t;
        t.v1 = TriangleCell::Point3(vertices[0][0]-x, vertices[0][1]-y, vertices[0][2]-z);
        t.v2 = TriangleCell::Point3(vertices[1][0]-x, vertices[1][1]-y, vertices[1][2]-z);
        t.v3 = TriangleCell::Point3(vertices[2][0]-x, vertices[2][1]-y, vertices[2][2]-z);
        int is_outside = triangleCellIntersection(t);
        if (is_outside == 0)
          voxels.push_back({x, y, z});
      }
    }
  }
}


void EuclideanViewMask::dilate(int iterations) {

}

}  // namespace MvsTexturing
