#define _USE_MATH_DEFINES
#include "tinyphash.hpp"
#include <memory>
#include <cmath>
#include <cstddef>
#include <array>
#include <cassert>
#include <algorithm>
#include <bitset>

typedef typename std::make_signed<size_t>::type ssize_t;

const size_t square = 32;
const size_t crop = 8;
const ssize_t smear_radius = 3;
const ssize_t smear_diameter = smear_radius * 2 + 1;
static_assert(crop < square);
static_assert(crop * crop % 2 == 0);


#ifdef HAVE_FREEIMAGE
#include <FreeImage.h>
struct BitmapReleaser{
	void operator()(FIBITMAP* bitmap) {
		if (bitmap)
			FreeImage_Unload(bitmap);
	}
};

typedef std::unique_ptr<FIBITMAP, BitmapReleaser> bitmap_ptr;
typedef uintptr_t ulongT;

bitmap_ptr convert_to_32bits(const bitmap_ptr &bitmap){
	bitmap_ptr converted(FreeImage_ConvertTo32Bits(bitmap.get()));
	if (converted)
		return converted;
	converted.reset(FreeImage_ConvertToType(bitmap.get(), FIT_BITMAP, true));
	if (!converted)
		return {};
	return bitmap_ptr(FreeImage_ConvertTo32Bits(converted.get()));
}

std::tuple<std::vector<std::uint8_t>, unsigned, unsigned> load_image_as_luma(const char *path){
	auto format = FreeImage_GetFileType(path);
	if (format == FIF_UNKNOWN)
		return {{}, 0, 0};
	bitmap_ptr bitmap(FreeImage_Load(format, path, format == FIF_JPEG ? JPEG_ACCURATE : 0));
	if (!bitmap)
		return {{}, 0, 0};
	bitmap = convert_to_32bits(bitmap);
	if (!bitmap)
		return {{}, 0, 0};

	auto width = FreeImage_GetWidth(bitmap.get());
	auto height = FreeImage_GetHeight(bitmap.get());
	std::vector<std::uint8_t> ret(width * height);
	for (unsigned y = 0; y < height; y++){
		auto row = FreeImage_GetScanLine(bitmap.get(), height - 1 - y);
		for (unsigned x = 0; x < width; x++){
			auto src = row + x * 4;
			float r = src[2];
			float g = src[1];
			float b = src[0];
			float a = src[3];

			r = r * a / 255;
			g = g * a / 255;
			b = b * a / 255;
			
			auto luma = (66 * r + 129 * g + 25 * b + 128) / 256 + 16;
			if (luma < 0)
				luma = 0;
			else if (luma > 255)
				luma = 255;
			ret[y * width + x] = (std::uint8_t)luma;
		}
	}
	return {ret, width, height};
}
#endif

namespace {

void box_blur(float *dst, const float *src, ssize_t width, ssize_t height, ssize_t stride, ssize_t pitch, ssize_t smear_radius){
	for (ssize_t y = 0; y < height; y++){
		auto src_row = src + y * pitch;
		auto dst_row = dst + y * pitch;
		for (ssize_t x = 0; x < width; x++){
			float accum = 0;
			for (auto i = -smear_radius; i <= smear_radius; i++){
				auto x0 = x + i;
				if (x0 < 0)
					x0 = 0;
				else if (x0 >= width)
					x0 = width - 1;
				accum += src_row[x0 * stride];
			}
			dst_row[x * stride] = accum;
		}
	}
}

std::vector<float> shrink_to_square(const std::vector<float> &image, unsigned size, unsigned width, unsigned height){
	std::vector<float> ret(size * size);
	for (unsigned y = 0; y < size; y++){
		for (unsigned x = 0; x < size; x++){
			auto x0 = width * x / size;
			auto y0 = height * y / size;
			ret[x + y * size] = image[x0 + y0 * width];
		}
	}
	return ret;
}

std::vector<float> smear_and_shrink(const std::uint8_t *bitmap, ssize_t width, ssize_t height, ssize_t smear_radius, ssize_t square){
	std::vector<float> ret(square * square);
	for (unsigned y = 0; y < square; y++){
		auto dst_row = ret.data() + y * square;
		for (unsigned x = 0; x < square; x++){
			auto x1 = width * x / square;
			auto y1 = height * y / square;
		
			float accum = 0;
			for (auto i = -smear_radius; i <= smear_radius; i++){
				auto y2 = (ssize_t)y1 + i;
				if (y2 < 0)
					y2 = 0;
				else if (y2 >= height)
					y2 = width - 1;
				auto src_row = bitmap + y2 * width;
				for (auto j = -smear_radius; j <= smear_radius; j++){
					auto x2 = (ssize_t)x1 + j;
					if (x2 < 0)
						x2 = 0;
					else if (x2 >= width)
						x2 = width - 1;
					accum += (float)src_row[x2];
				}
			}
			dst_row[x] = accum;
		}
	}
	return ret;
}

void matrix_multiplication(std::vector<float> &dst, const std::vector<float> &left, const std::vector<float> &right, size_t size){
	assert(left.size() == right.size());
	assert(left.size() == dst.size());
	assert(left.size() == size * size);
	auto p = dst.data();
	for (size_t y = 0; y < size; y++){
		for (size_t x = 0; x < size; x++){
			double accum = 0;
			for (size_t i = 0; i < size; i++)
				accum += left[i + y * size] * right[x + i * size];
			*(p++) = (float)accum;
		}
	}
}

}

TinyPHash::TinyPHash(){
	this->matrix.resize(square * square, 1 / sqrt((float)square));
	this->matrix_transpose = this->matrix;
    auto c1 = (float)sqrt(2.0 / square);
	auto m = (M_PI / 2 / square);
	for (size_t y = 1; y < square; y++){
		auto p = &this->matrix[y * square];
		for (size_t x = 0; x < square; x++){
			auto v = c1 * (float)cos(m * y * (double)(2 * x + 1));
			this->matrix_transpose[y + x * square] = *(p++) = v;
		}
	}
}

std::uint64_t TinyPHash::dct_imagehash(const void *void_bitmap, unsigned width, unsigned height) const{
	std::vector<float> temp;
	auto bitmap = (const std::uint8_t *)void_bitmap;
	
	if (width >= square * smear_diameter && height >= square * smear_diameter){
		//When the image is at least 224 pixels in both dimensions, the values of the
		//pixels in the shrunken image are affected only by the pixels in a 7x7 square
		//around the pixel in the original image, so it is not necessary to blur most
		//of the images. This optimization means the function can be made to take O(1)
		//time over the size of the bitmap.
		temp = smear_and_shrink(bitmap, width, height, smear_radius, square);
	}else{
		temp.resize(width * height);
		std::copy(bitmap, bitmap + temp.size(), temp.begin());

		//7x7 box blur. Blur in one dimension then in the other.
		std::vector<float> temp2(width * height);
		box_blur(temp2.data(), temp .data(), width , height,     1, width, smear_radius);
		box_blur(temp .data(), temp2.data(), height, width , width,     1, smear_radius);

		temp = shrink_to_square(temp, square, width, height);
	}

	//Compute the discrete cosine transform of temp.
	{
		std::vector<float> temp2(square * square);
		matrix_multiplication(temp2, this->matrix, temp                  , square);
		matrix_multiplication(temp , temp2       , this->matrix_transpose, square);
	}

	//Eliminate the very lowest and the high frequencies.
	{
		std::vector<float> temp2(crop * crop);
		for (size_t y = 0; y < crop; y++)
			for (size_t x = 0; x < crop; x++)
				temp2[x + y * crop] = temp[1 + x + (1 + y) * square];
		temp = std::move(temp2);
	}

	//Compute the median.
	float median;
	{
		auto temp2 = temp;
		std::sort(temp2.begin(), temp2.end());
		auto n = temp2.size() / 2;
		median = (temp2[n] + temp2[n - 1]) / 2;
	}

	std::uint64_t ret = 0;
	for (int i = (int)std::min<size_t>(crop * crop, 64); i--;){
		auto x = temp[i];
		ret <<= 1;
		ret |= x > median;
	}
	
	return ret;
}

std::uint64_t tinyph_dct_imagehash(const void *void_bitmap, unsigned width, unsigned height){
	return TinyPHash().dct_imagehash(void_bitmap, width, height);
}

extern "C" void *allocate_tinyphash(){
	return new TinyPHash();
}

extern "C" int tinyph_dct_imagehash_iterated(
			uint64_t *hash,
			const void *tinyphash,
			const void *bitmap,
			unsigned width,
			unsigned height
		){
	try{
		*hash = static_cast<const TinyPHash *>(tinyphash)->dct_imagehash(bitmap, width, height);
	}catch (std::bad_alloc &){
		return 0;
	}
	return 1;
}

extern "C" int tinyph_dct_imagehash(uint64_t *hash, const void *bitmap, unsigned width, unsigned height){
	try{
		*hash = tinyph_dct_imagehash(bitmap, width, height);
	}catch (std::bad_alloc &){
		return 0;
	}
	return 1;
}

extern "C" int tinyph_hamming_distance(uint64_t a, uint64_t b){
#if defined _MSC_VER
	return (int)__popcnt64(a ^ b);
#elif defined __GNUC__ || defined __clang__
	return (int)__builtin_popcountll(a ^ b);
#else
	return (int)std::bitset<64>(a ^ b).count();
#endif
}
