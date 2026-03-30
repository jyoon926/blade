#include "blade/data/mnist.h"
#include <fstream>
#include <stdexcept>
#include <string>

namespace blade {
namespace data {

uint32_t MNIST::read_uint32_be(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8)  |  uint32_t(b[3]);
}

void MNIST::load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("MNIST: cannot open " + path);

    uint32_t magic = read_uint32_be(f);
    if (magic != 0x00000803)
        throw std::runtime_error("MNIST: bad image file magic");

    uint32_t n     = read_uint32_be(f);
    uint32_t rows  = read_uint32_be(f);
    uint32_t cols  = read_uint32_be(f);

    images_.reserve(n);
    std::vector<uint8_t> buf(rows * cols);
    for (uint32_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(buf.data()), rows * cols);
        Tensor img({1, rows, cols});
        float* p = img.data_ptr();
        for (size_t j = 0; j < rows * cols; ++j)
            p[j] = buf[j] / 255.f;
        images_.push_back(std::move(img));
    }
}

void MNIST::load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("MNIST: cannot open " + path);

    uint32_t magic = read_uint32_be(f);
    if (magic != 0x00000801)
        throw std::runtime_error("MNIST: bad label file magic");

    uint32_t n = read_uint32_be(f);
    labels_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t label;
        f.read(reinterpret_cast<char*>(&label), 1);
        labels_[i] = (int)label;
    }
}

MNIST::MNIST(const std::string& root, Split split, bool download) {
    (void)download;  // TODO: download support
    std::string prefix = root + "/";
    if (split == Split::Train) {
        load_images(prefix + "train-images-idx3-ubyte");
        load_labels(prefix + "train-labels-idx1-ubyte");
    } else {
        load_images(prefix + "t10k-images-idx3-ubyte");
        load_labels(prefix + "t10k-labels-idx1-ubyte");
    }
}

size_t MNIST::size() const { return images_.size(); }

Sample MNIST::get(size_t idx) const {
    Tensor label_t({1});
    label_t.data_ptr()[0] = (float)labels_[idx];
    return {images_[idx], label_t};
}

} // namespace data
} // namespace blade
