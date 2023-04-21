#pragma once

#ifdef __cplusplus
#include <vector>
#include <cstdint>

#ifdef HAVE_FREEIMAGE
#include <tuple>

std::tuple<std::vector<std::uint8_t>, unsigned, unsigned> load_image_as_luma(const char *path);
#endif

class TinyPHash{
	std::vector<float> matrix;
	std::vector<float> matrix_transpose;
public:
	TinyPHash();
	TinyPHash(const TinyPHash &) = default;
	TinyPHash &operator=(const TinyPHash &) = default;
	TinyPHash(TinyPHash &&other) = default;
	TinyPHash &operator=(TinyPHash &&other) = default;
	std::uint64_t dct_imagehash(const void *bitmap, unsigned width, unsigned height) const;
};

std::uint64_t tinyph_dct_imagehash(const void *bitmap, unsigned width, unsigned height);

#define EXTERN_C extern "C"
#else
#include <stdint.h>

#define EXTERN_C
#endif

EXTERN_C void *allocate_tinyphash();
EXTERN_C int tinyph_dct_imagehash_iterated(
	uint64_t *hash,
	const void *tinyphash,
	const void *bitmap,
	unsigned width,
	unsigned height
);
EXTERN_C int tinyph_dct_imagehash(
	uint64_t *hash,
	const void *bitmap,
	unsigned width,
	unsigned height
);
EXTERN_C int tinyph_hamming_distance(uint64_t, uint64_t);
