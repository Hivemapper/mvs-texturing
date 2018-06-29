#include "wrapper/wrapper.h"

#include "tex/util.h"

namespace MvsTexturing {

std::string testFunc(int n) {
  return "BEHOLD: " + std::to_string(n) + number_suffix(n);
}

}  // namespace MvsTexturing
