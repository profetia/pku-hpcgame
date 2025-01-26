#include <openssl/evp.h>
#include <openssl/sha.h>
#include <secp256k1.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

std::string toHex(const uint8_t* data, size_t size) {
  std::ostringstream oss;
  for (size_t i = 0; i < size; ++i) {
    oss << std::hex << std::setfill('0') << std::setw(2) << (int)data[i];
  }
  return oss.str();
}

constexpr int NUM_THREADS = 8;
constexpr int NUM_PREFIXES = 3;
constexpr int PREFIX_LENGTH = 3;
constexpr int PREFIX_OFFSET = 12;

struct Solution {
  uint8_t private_key[32];
  uint8_t ethereum_address[32];
};

using Query = uint8_t[PREFIX_LENGTH];

static __always_inline bool prefix_matches(const uint8_t* query,
                                           const uint8_t* address) {
  return query[0] == address[PREFIX_OFFSET] &&
         query[1] == address[PREFIX_OFFSET + 1] &&
         query[2] == address[PREFIX_OFFSET + 2];
}

class VanitySolver {
  std::unique_ptr<Query[]> queries_;

  std::atomic<int> num_solved_;
  std::unique_ptr<std::atomic<Solution*>[]> solutions_;

 public:
  VanitySolver(std::unique_ptr<Query[]> queries)
      : queries_(std::move(queries)),
        num_solved_(0),
        solutions_(new std::atomic<Solution*>[NUM_PREFIXES] {}) {}

  bool try_solve(Solution* solution) {
    for (int i = 0; i < NUM_PREFIXES; ++i) {
      if (solutions_[i].load() != nullptr) {
        continue;
      }

      if (!prefix_matches(queries_[i], solution->ethereum_address)) {
        continue;
      }

      Solution* expected = nullptr;
      if (solutions_[i].compare_exchange_strong(expected, solution)) {
        num_solved_++;
        return true;
      }
    }
    return false;
  }

  bool is_done() const { return num_solved_ == NUM_PREFIXES; }

  void finalize(std::ostream& out) {
    for (int i = 0; i < NUM_PREFIXES; ++i) {
      Solution* solution = solutions_[i].load();

      std::string address =
          toHex(solution->ethereum_address + PREFIX_OFFSET, 32 - PREFIX_OFFSET);
      std::string private_key = toHex(solution->private_key, 32);

      out << "0x" << address << "\n" << private_key << "\n";
    }
  }

  ~VanitySolver() {
    for (int i = 0; i < NUM_PREFIXES; ++i) {
      delete solutions_[i].load();
    }
  }
};

static __always_inline void generate_random_private_key(FILE* urandom,
                                                        uint8_t* private_key) {
  fread(private_key, 1, 32, urandom);
}

static __always_inline void sha3256(EVP_MD_CTX* context, const EVP_MD* md,
                                    const uint8_t* data, size_t size,
                                    uint8_t* hash) {
  EVP_MD_CTX_init(context);
  EVP_DigestInit_ex(context, md, nullptr);
  EVP_DigestUpdate(context, data, size);

  unsigned int hashLen;
  EVP_DigestFinal_ex(context, hash, &hashLen);
}

static __always_inline void compute_ethereum_address(
    secp256k1_context* ctx, secp256k1_pubkey* pubkey,
    uint8_t* pubkey_serialized, size_t* pubkey_serialized_len,
    EVP_MD_CTX* context, const EVP_MD* md, const uint8_t* private_key,
    uint8_t* ethereum_address) {
  secp256k1_ec_pubkey_create(ctx, pubkey, private_key);
  secp256k1_ec_pubkey_serialize(ctx, pubkey_serialized, pubkey_serialized_len,
                                pubkey, SECP256K1_EC_UNCOMPRESSED);
  sha3256(context, md, pubkey_serialized + 1, *pubkey_serialized_len - 1,
          ethereum_address);
}

void compute_vanity(VanitySolver* solver) {
  secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
  FILE* urandom = fopen("/dev/urandom", "rb");
  secp256k1_pubkey pubkey;
  uint8_t pubkey_serialized[65];
  size_t pubkey_serialized_len = 65;
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  const EVP_MD* md = EVP_get_digestbyname("sha3-256");

  Solution* solution = new Solution{};
  while (!solver->is_done()) {
    generate_random_private_key(urandom, solution->private_key);
    compute_ethereum_address(ctx, &pubkey, pubkey_serialized,
                             &pubkey_serialized_len, context, md,
                             solution->private_key, solution->ethereum_address);

    if (solver->try_solve(solution)) {
      solution = new Solution{};
    }
  }

  delete solution;
  EVP_MD_CTX_free(context);
  fclose(urandom);
  secp256k1_context_destroy(ctx);
}

int main(int argc, char* argv[]) {
  std::ifstream infile("vanity.in");

  std::unique_ptr<Query[]> queries(new Query[NUM_PREFIXES]);

  std::string line;
  for (int i = 0; i < NUM_PREFIXES; ++i) {
    infile >> line;

    uint32_t prefix = std::stoul(line, nullptr, 16);
    queries[i][0] = (prefix >> 16) & 0xff;
    queries[i][1] = (prefix >> 8) & 0xff;
    queries[i][2] = prefix & 0xff;
  }

  VanitySolver solver(std::move(queries));

  std::thread threads[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads[i] = std::thread(compute_vanity, &solver);
  }

  for (int i = 0; i < NUM_THREADS; ++i) {
    threads[i].join();
  }

  std::ofstream outfile("vanity.out");
  solver.finalize(outfile);

  return 0;
}