#pragma once

#include <vector>
#include <map>
#include <set>
#include <Eigen/Geometry>

namespace MvsTexturing {


class EuclideanViewMask {
 public:
  EuclideanViewMask(const Eigen::Matrix<double, 3, 1>& vmin,
                    const Eigen::Matrix<double, 3, 3>& coord_transform,
                    int nx,
                    int ny);

  bool isValidXy(int x, int y) const;
  std::vector<int> getVoxelIndex(const Eigen::Matrix<double, 3, 1>& v) const;
  bool isValidVector(const Eigen::Matrix<double, 3, 1>& v) const;

  const std::set<uint16_t>& operator[](const std::vector<int>& xyz) const;
  bool contains(const std::vector<int>& xyz, int i) const;
  // const std::set<uint16_t>& operator[](int x, int y, int z) const;

  void insert(const Eigen::Matrix<double, 3, 1>& v, uint16_t i);
  void insert(const Eigen::Matrix<double, 3, 1>& v, const std::set<uint16_t>& is);

  void dilate(int iterations);

  void getTriangleVoxels(const std::vector<Eigen::Matrix<double, 3, 1>>& vertices,
                         std::vector<std::vector<int>>& voxels) const;

  int countCells() const;


  void convertToPoints(std::vector<Eigen::Matrix<double, 3, 1>>& points, int cell_subdivisions = 0) const;

  Eigen::Matrix<double, 3, 1> getCellSize() const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  Eigen::Matrix<double, 3, 1> vmin;
  Eigen::Matrix<double, 3, 3> coord_transform;

  std::vector<std::vector<std::map<int, std::set<uint16_t>>>> mask_data;

  std::set<uint16_t>& get(const std::vector<int>& xyz);
};

}  // namespace MvsTexturing
