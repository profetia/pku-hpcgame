#include <mpi.h>

#include <fstream>
#include <iostream>
#include <vector>

using Cell = uint8_t;
constexpr Cell EMPTY = 0 + '0';
constexpr Cell TREE = 1 + '0';
constexpr Cell FIRE = 2 + '0';
constexpr Cell ASH = 3 + '0';

using Event = uint16_t;
constexpr Event THUNDER = 1;
constexpr Event REBIRTH = 2;

using Time = uint16_t;
using Index = uint16_t;

using CommRank = int;
using FileOffset = int32_t;

struct Fight {
  Time ts;
  Event type;
  Index x1, y1;
  Index x2, y2;
};

static __always_inline void apply_fight(Cell *forest_local, const Fight *fight,
                                        Index kN, Index kRank, Index kChunkSize,
                                        Index kChunkOffset) {
  if (fight->type == THUNDER) {
    Index x = fight->x1;
    Index y = fight->y1;

    if (x >= kChunkOffset && x < kChunkOffset + kChunkSize) {
      Index local_x = x - kChunkOffset + 1;

      if (forest_local[local_x * kN + y] == TREE) {
        forest_local[local_x * kN + y] = FIRE;
      }
    }
  } else if (fight->type == REBIRTH) {
    Index x1 = fight->x1;
    Index y1 = fight->y1;
    Index x2 = fight->x2;
    Index y2 = fight->y2;

    if (x1 < kChunkOffset + kChunkSize && x2 >= kChunkOffset) {
      Index local_x1 = x1 < kChunkOffset ? 1 : x1 - kChunkOffset + 1;
      Index local_x2 =
          x2 >= kChunkOffset + kChunkSize ? kChunkSize : x2 - kChunkOffset + 1;

      for (Index i = local_x1; i <= local_x2; ++i) {
        for (Index j = y1; j <= y2; ++j) {
          if (forest_local[i * kN + j] == ASH) {
            forest_local[i * kN + j] = TREE;
          }
        }
      }
    }
  }
}

static __always_inline bool is_top_fire(const Cell *forest_local, Index kN,
                                        Index kRank, Index kSize,
                                        Index kChunkSize, Index i, Index j) {
  if (kRank == 0 && i == 1) {
    return false;
  }

  return forest_local[(i - 1) * kN + j] == FIRE;
}

static __always_inline bool is_bottom_fire(const Cell *forest_local, Index kN,
                                           Index kRank, Index kSize,
                                           Index kChunkSize, Index i, Index j) {
  if (kRank == kSize - 1 && i == kChunkSize) {
    return false;
  }

  return forest_local[(i + 1) * kN + j] == FIRE;
}

static __always_inline bool is_left_fire(const Cell *forest_local, Index kN,
                                         Index kRank, Index kSize,
                                         Index kChunkSize, Index i, Index j) {
  if (j == 0) {
    return false;
  }

  return forest_local[i * kN + j - 1] == FIRE;
}

static __always_inline bool is_right_fire(const Cell *forest_local, Index kN,
                                          Index kRank, Index kSize,
                                          Index kChunkSize, Index i, Index j) {
  if (j == kN - 1) {
    return false;
  }

  return forest_local[i * kN + j + 1] == FIRE;
}

static __always_inline bool should_be_fire(const Cell *forest_local, Index kN,
                                           Index kRank, Index kSize,
                                           Index kChunkSize, Index i, Index j) {
  if (forest_local[i * kN + j] != TREE) {
    return false;
  }

  return is_top_fire(forest_local, kN, kRank, kSize, kChunkSize, i, j) ||
         is_bottom_fire(forest_local, kN, kRank, kSize, kChunkSize, i, j) ||
         is_left_fire(forest_local, kN, kRank, kSize, kChunkSize, i, j) ||
         is_right_fire(forest_local, kN, kRank, kSize, kChunkSize, i, j);
}

static __always_inline bool should_be_ash(const Cell *forest_local, Index kN,
                                          Index kRank, Index kSize,
                                          Index kChunkSize, Index i, Index j) {
  return forest_local[i * kN + j] == FIRE;
}

static __always_inline void update_forest(const Cell *forest_local, Index kN,
                                          Index kRank, Index kSize,
                                          Index kChunkSize,
                                          Cell *new_forest_local) {
  for (Index i = 1; i <= kChunkSize; ++i) {
    for (Index j = 0; j < kN; ++j) {
      if (should_be_fire(forest_local, kN, kRank, kSize, kChunkSize, i, j)) {
        new_forest_local[i * kN + j] = FIRE;
      } else if (should_be_ash(forest_local, kN, kRank, kSize, kChunkSize, i,
                               j)) {
        new_forest_local[i * kN + j] = ASH;
      } else {
        new_forest_local[i * kN + j] = forest_local[i * kN + j];
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>"
              << std::endl;
    return 1;
  }
  const char *kInputFile = argv[1];
  const char *kOutputFile = argv[2];

  MPI_Init(&argc, &argv);

  CommRank kRank, kSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
  MPI_Comm_size(MPI_COMM_WORLD, &kSize);

  Index kN, kM;
  Time kT;

  FileOffset header_offset;
  Fight *fights;

  {
    std::ifstream in;
    if (kRank == 0) {
      in.open(kInputFile);

      in >> kN >> kM >> kT;
      header_offset = static_cast<FileOffset>(in.tellg()) + 1;
    }

    MPI_Bcast(&kN, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kM, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kT, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);

    MPI_Bcast(&header_offset, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

    MPI_Alloc_mem(kM * sizeof(Fight), MPI_INFO_NULL, &fights);
    in.seekg(header_offset + kN * kN * sizeof(char) * 2);

    for (Index i = 0; i < kM; ++i) {
      Fight &fight = fights[i];
      in >> fight.ts >> fight.type;
      if (fight.type == THUNDER) {
        in >> fight.x1 >> fight.y1;
      } else if (fight.type == REBIRTH) {
        in >> fight.x1 >> fight.y1 >> fight.x2 >> fight.y2;
      }
    }

    MPI_Bcast(fights, kM * sizeof(Fight), MPI_BYTE, 0, MPI_COMM_WORLD);
  }

  Index kChunkSize = kN / kSize;
  Index kChunkOffset = kRank * kChunkSize;
  Cell *forest_local, *new_forest_local;
  MPI_Alloc_mem((2 + kChunkSize) * kN * sizeof(Cell), MPI_INFO_NULL,
                &forest_local);
  MPI_Alloc_mem((2 + kChunkSize) * kN * sizeof(Cell), MPI_INFO_NULL,
                &new_forest_local);

  {
    FileOffset offset =
        header_offset + kRank * kChunkSize * kN * sizeof(char) * 2;

    MPI_Datatype cell_type, cell_space_type;
    MPI_Type_contiguous(sizeof(Cell), MPI_BYTE, &cell_type);
    MPI_Type_create_resized(cell_type, 0, 2 * sizeof(Cell), &cell_space_type);
    MPI_Type_commit(&cell_space_type);

    MPI_File in;
    MPI_File_open(MPI_COMM_WORLD, kInputFile, MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &in);

    MPI_File_set_view(in, offset, MPI_BYTE, cell_space_type, "native",
                      MPI_INFO_NULL);

    MPI_File_read_all(in, forest_local + kN, kChunkSize * kN, MPI_BYTE,
                      MPI_STATUS_IGNORE);

    MPI_File_close(&in);
  }

  CommRank left = (kRank == 0) ? MPI_PROC_NULL : kRank - 1;
  CommRank right = (kRank == kSize - 1) ? MPI_PROC_NULL : kRank + 1;

  Fight *next_fight = fights;
  for (Time step = 1; step <= kT; ++step) {
    if (next_fight != fights + kM && next_fight->ts == step) {
      apply_fight(forest_local, next_fight, kN, kRank, kChunkSize,
                  kChunkOffset);
      ++next_fight;
    }

    MPI_Sendrecv(forest_local + kN, kN, MPI_UINT8_T, left, 0,
                 forest_local + (kChunkSize + 1) * kN, kN, MPI_UINT8_T, right,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(forest_local + kChunkSize * kN, kN, MPI_UINT8_T, right, 0,
                 forest_local, kN, MPI_UINT8_T, left, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    update_forest(forest_local, kN, kRank, kSize, kChunkSize, new_forest_local);

    std::swap(forest_local, new_forest_local);
  }

  {
    FileOffset kLineSize = (kN * 2 + 1) * sizeof(char);
    FileOffset offset = kRank * kChunkSize * kLineSize;

    char *line;
    MPI_Alloc_mem(kLineSize, MPI_INFO_NULL, &line);

    MPI_File out;
    MPI_File_open(MPI_COMM_WORLD, kOutputFile,
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &out);

    for (Index i = 1; i <= kChunkSize; ++i) {
      for (Index j = 0; j < kN; ++j) {
        line[j * 2] = forest_local[i * kN + j];
        line[j * 2 + 1] = ' ';
      }
      line[kN * 2] = '\n';

      MPI_File_write_at_all(out, offset, line, kN * 2 + 1, MPI_BYTE,
                            MPI_STATUS_IGNORE);

      offset += kLineSize;
    }

    MPI_File_close(&out);
    MPI_Free_mem(line);
  }

  MPI_Free_mem(new_forest_local);
  MPI_Free_mem(forest_local);
  MPI_Free_mem(fights);

  MPI_Finalize();

  return 0;
}