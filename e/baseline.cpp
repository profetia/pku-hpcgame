#include <mpi.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using Cell = uint16_t;
constexpr Cell EMPTY = 0;
constexpr Cell TREE = 1;
constexpr Cell FIRE = 2;
constexpr Cell ASH = 3;

using Event = uint16_t;
constexpr Event THUNDER = 1;
constexpr Event REBIRTH = 2;

using Time = uint16_t;
using Index = uint16_t;

struct Fight {
  Time ts;
  Event type;
  Index x1, y1;
  Index x2, y2;
};

static void apply_fight(Cell *forest, const Fight *fight, Index n) {
  if (fight->type == THUNDER) {
    Index x = fight->x1;
    Index y = fight->y1;
    if (forest[x * n + y] == TREE) {
      forest[x * n + y] = FIRE;
    }
  } else if (fight->type == REBIRTH) {
    Index x1 = fight->x1;
    Index y1 = fight->y1;
    Index x2 = fight->x2;
    Index y2 = fight->y2;
    for (Index i = x1; i <= x2; ++i) {
      for (Index j = y1; j <= y2; ++j) {
        if (forest[i * n + j] == ASH) {
          forest[i * n + j] = TREE;
        }
      }
    }
  }
}

static bool is_top_fire(const Cell *forest, Index n, Index i, Index j) {
  if (i == 0) {
    return false;
  }
  return forest[(i - 1) * n + j] == FIRE;
}

static bool is_bottom_fire(const Cell *forest, Index n, Index i, Index j) {
  if (i == n - 1) {
    return false;
  }
  return forest[(i + 1) * n + j] == FIRE;
}

static bool is_left_fire(const Cell *forest, Index n, Index i, Index j) {
  if (j == 0) {
    return false;
  }
  return forest[i * n + j - 1] == FIRE;
}

static bool is_right_fire(const Cell *forest, Index n, Index i, Index j) {
  if (j == n - 1) {
    return false;
  }
  return forest[i * n + j + 1] == FIRE;
}

static bool should_be_fire(const Cell *forest, Index n, Index i, Index j) {
  if (forest[i * n + j] != TREE) {
    return false;
  }

  return is_top_fire(forest, n, i, j) || is_bottom_fire(forest, n, i, j) ||
         is_left_fire(forest, n, i, j) || is_right_fire(forest, n, i, j);
}

static bool should_be_ash(const Cell *forest, Index n, Index i, Index j) {
  return forest[i * n + j] == FIRE;
}

static void update_forest(const Cell *forest, Index n, Cell *new_forest) {
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      if (should_be_fire(forest, n, i, j)) {
        new_forest[i * n + j] = FIRE;
      } else if (should_be_ash(forest, n, i, j)) {
        new_forest[i * n + j] = ASH;
      } else {
        new_forest[i * n + j] = forest[i * n + j];
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
  const char *input_file = argv[1];
  const char *output_file = argv[2];

  std::ifstream in(input_file);

  Index n, m;
  Time t;
  in >> n >> m >> t;

  std::unique_ptr<Cell[]> forest(new Cell[n * n]);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      in >> forest[i * n + j];
    }
  }

  std::unique_ptr<Fight[]> fights(new Fight[m]);
  for (Time i = 0; i < m; ++i) {
    Fight &fight = fights[i];
    in >> fight.ts >> fight.type;
    if (fight.type == THUNDER) {
      in >> fight.x1 >> fight.y1;
    } else if (fight.type == REBIRTH) {
      in >> fight.x1 >> fight.y1 >> fight.x2 >> fight.y2;
    }
  }

  in.close();

  Fight *next_fight = fights.get();
  std::unique_ptr<Cell[]> new_forest(new Cell[n * n]);

  for (Time step = 1; step <= t; ++step) {
    if (step == next_fight->ts) {
      apply_fight(forest.get(), next_fight, n);
      ++next_fight;
    }

    update_forest(forest.get(), n, new_forest.get());

    std::swap(forest, new_forest);
  }

  std::ofstream out(output_file);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      out << forest[i * n + j] << " ";
    }
    out << std::endl;
  }

  return 0;
}