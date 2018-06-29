#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <Eigen/Geometry>

namespace MvsTexturing {


class EuclideanViewMask {
 public:
  EuclideanViewMask();
  EuclideanViewMask(const Eigen::Matrix<double, 3, 1>& vmin,
                    const Eigen::Matrix<double, 3, 3>& coord_transform,
                    int nx,
                    int ny);

  bool isValidXy(int x, int y) const;
  std::vector<int> getVoxelIndex(const Eigen::Matrix<double, 3, 1>& v) const;

  const std::set<uint16_t>& operator[](const std::vector<int>& xyz) const;
  // const std::set<uint16_t>& operator[](int x, int y, int z) const;

  void append(const Eigen::Matrix<double, 3, 1>& v, uint16_t i);
  void append(const Eigen::Matrix<double, 3, 1>& v, const std::set<uint16_t>& is);

  void dilate(int iterations);

  void getTriangleVoxels(const std::vector<Eigen::Matrix<double, 3, 1>>& vertices,
                         std::vector<std::vector<int>>& voxels) const;
 private:
  Eigen::Matrix<double, 3, 1> vmin;
  Eigen::Matrix<double, 3, 3> coord_transform;
  std::string name = "EVMask";

  std::vector<std::vector<std::map<int, std::set<uint16_t>>>> mask_data;

  std::set<uint16_t>& get(const std::vector<int>& xyz);
};

}  // namespace MvsTexturing
