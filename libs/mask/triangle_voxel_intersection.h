#pragma once

namespace TriangleCell {

struct Point3 {
  float           x;
  float           y;
  float           z;
  Point3() {x = y = z = 0.0;}
  Point3(double xd, double yd, double zd) {x = xd; y = yd; z = zd;}
};

struct Triangle3{
  Point3 v1;
  Point3 v2;
  Point3 v3;
};

// Returns true of t intersects the unit cube *centered* at the origin
int triangleCellIntersection(Triangle3 t);

}  // namespace TriangleCell
