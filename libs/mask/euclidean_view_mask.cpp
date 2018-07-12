#include "mask/euclidean_view_mask.h"

#include <exception>
#include <iostream>
#include <string>

#include "mask/triangle_voxel_intersection.h"

using std::vector;
using std::set;

namespace MvsTexturing {

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
 * @brief Returns true if the supplied vector is within the bounds of the grid
 */
bool EuclideanViewMask::isValidVector(const Eigen::Matrix<double, 3, 1>& v) const {
  Eigen::Matrix<double, 3, 1> vi = coord_transform*(v - vmin);
  return isValidXy(floor(vi[0]), floor(vi[1]));
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
    throw "Warning: location results in invalid coordinates";
  }
  return xyz;
}

/**
 * @brief (private) non const accessor/mutator
 */
set<uint16_t>& EuclideanViewMask::get(const vector<int>& xyz) {
  if (!isValidXy(xyz[0], xyz[1])) {
    throw "invalid coordinates " + std::to_string(xyz[0]) + " " + std::to_string(xyz[1]);
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

/**
 * @brief const membership checker
 */
bool EuclideanViewMask::contains(const vector<int>& xyz, int i) const {
  if (isValidXy(xyz[0], xyz[1])) {
    if (mask_data[xyz[0]][xyz[1]].count(xyz[2]))
      return mask_data[xyz[0]][xyz[1]].at(xyz[2]).count(i);
    else
      return false;
  } else {
    return false;
  }
}


void EuclideanViewMask::insert(const Eigen::Matrix<double, 3, 1>& v, uint16_t i) {
  set<uint16_t>& voxel = get(getVoxelIndex(v));
  voxel.insert(i);
}

void EuclideanViewMask::insert(const Eigen::Matrix<double, 3, 1>& v,  const set<uint16_t>& is) {
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
  vector<int> mins(3, INT_MAX);
  vector<int> maxes(3, -INT_MIN);
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
        t.v1 = TriangleCell::Point3(vertices[0][0]-x-0.5, vertices[0][1]-y-0.5, vertices[0][2]-z-0.5);
        t.v2 = TriangleCell::Point3(vertices[1][0]-x-0.5, vertices[1][1]-y-0.5, vertices[1][2]-z-0.5);
        t.v3 = TriangleCell::Point3(vertices[2][0]-x-0.5, vertices[2][1]-y-0.5, vertices[2][2]-z-0.5);
        int is_outside = triangleCellIntersection(t);
        if (is_outside == 0)
          voxels.push_back({x, y, z});
      }
    }
  }
}

int EuclideanViewMask::countCells() const {
  int sum = 0;
  for (const auto& column : mask_data) {
    for (const auto& stack : column) {
      sum += stack.size();
    }
  }
  return sum;
}

Eigen::Matrix<double, 3, 1> EuclideanViewMask::getCellSize() const {
  return coord_transform.inverse()*Eigen::Matrix<double, 3, 1>(1.0, 1.0, 1.0);
}


void EuclideanViewMask::convertToPoints(std::vector<Eigen::Matrix<double, 3, 1>>& points, int cell_subdivisions) const {
  int nx = mask_data.size();
  int ny = mask_data[0].size();
  points.clear();
  Eigen::Matrix<double, 3, 1> cell_size = getCellSize();
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (const auto& cell : mask_data[i][j]) {
        points.push_back(Eigen::Matrix<double, 3, 1>(cell_size[0]*i + vmin[0],
                                                     cell_size[1]*j + vmin[1],
                                                     cell_size[2]*cell.first + vmin[2]));

      }
    }
  }
}

void EuclideanViewMask::dilate(int iterations) {
  int nx = mask_data.size();
  int ny = mask_data[0].size();
  int new_nx = nx + 2*iterations;
  int new_ny = ny + 2*iterations;

  std::vector<std::vector<std::map<int, std::set<uint16_t>>>> new_mask_data;
  new_mask_data.resize(new_nx);
  for (int i = 0; i < new_nx; ++i) {
    new_mask_data[i].resize(new_ny);
  }

  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (const auto& cell : mask_data[i][j]) {
        for (int ii = 0; ii <= 2*iterations; ++ii) {
          for (int jj = 0; jj <= 2*iterations; ++jj) {
            for (int kk = 0; kk <= 2*iterations; ++kk) {
              for (uint16_t frame : cell.second) {
                new_mask_data[i+ii][j+jj][cell.first + kk].insert(frame);
              }
            }
          }
        }
      }
    }
  }

  mask_data = new_mask_data;
  vmin = vmin - iterations*getCellSize();
}

}  // namespace MvsTexturing
