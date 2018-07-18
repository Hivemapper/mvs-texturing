#pragma once

#include <vector>
#include <map>
#include <set>
#include <utility>
#include <Eigen/Geometry>

namespace MvsTexturing {

// typedef std::pair<uint16_t, uint16_t> FrameRange;
class FrameRange {
 public:
  uint16_t first;
  uint16_t second;
  FrameRange(uint16_t f, uint16_t s) : first(f), second(s) {}
  bool operator<(const FrameRange& r) const {return second < r.first;}
};

bool rangesContain(const std::set<FrameRange>& ranges, uint16_t i);
void insertRange(std::set<FrameRange>& ranges, const FrameRange& ij);


class EuclideanViewMask {
 public:
  EuclideanViewMask(const Eigen::Matrix<double, 3, 1>& vmin,
                    const Eigen::Matrix<double, 3, 3>& coord_transform,
                    int nx,
                    int ny);

  bool isValidXy(int x, int y) const;
  std::vector<int> getVoxelIndex(const Eigen::Matrix<double, 3, 1>& v) const;
  bool isValidVector(const Eigen::Matrix<double, 3, 1>& v) const;

  const std::set<FrameRange>& operator[](const std::vector<int>& xyz) const;
  bool contains(const std::vector<int>& xyz, uint16_t i) const;
  bool contains(const Eigen::Matrix<double, 3, 1>& v, uint16_t i) const;
  // const std::set<uint16_t>& operator[](int x, int y, int z) const;

  void insert(const Eigen::Matrix<double, 3, 1>& v, uint16_t i);
  void insert(const Eigen::Matrix<double, 3, 1>& v, FrameRange range);
  void insert(const Eigen::Matrix<double, 3, 1>& v, const std::set<FrameRange>& ranges);

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

  std::vector<std::vector<std::map<int, std::set<FrameRange>>>> mask_data;

  std::set<FrameRange>& get(const std::vector<int>& xyz);
};

}  // namespace MvsTexturing
