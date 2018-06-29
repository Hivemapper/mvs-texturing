#include "texture_wrapper/wrapper.h"

#include "tex/util.h"

namespace TextureWrapper {

std::string testFunc(int n) {
  return "BEHOLD: " + std::to_string(n) + number_suffix(n);
  // return "foo";
}

}
