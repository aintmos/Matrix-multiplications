constexpr int rowNumA = 1024;      // A's row size
constexpr int colNumA = 1024;       // A's col size
constexpr int rowNumB = colNumA;    // B's row size
constexpr int colNumB = 1024;       // B's col Size
constexpr int rowNumBTP = colNumB;  // B transposed row size
constexpr int colNumBTP = rowNumB;  // B transposed col size
constexpr int rowNumC = rowNumA;    // C's row size
constexpr int colNumC = colNumB;    // C's col size
constexpr int subBlockSize = 32;

using dataType = int;