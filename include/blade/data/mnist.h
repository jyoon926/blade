#pragma once
#include "blade/data/dataset.h"
#include <string>

namespace blade {
namespace data {

// Reads the IDX binary format used by the MNIST dataset.
// Tensors are normalised to [0, 1] float32.
// Labels are integer class indices (0-9) stored as float for uniformity.
class MNIST : public Dataset {
public:
    enum class Split { Train, Test };

    // root: directory containing the four raw MNIST files
    // (train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.)
    MNIST(const std::string& root, Split split = Split::Train, bool download = false);

    size_t size() const override;
    Sample get(size_t idx) const override;

private:
    std::vector<Tensor> images_;  // each (1, 28, 28)
    std::vector<int>    labels_;

    void load_images(const std::string& path);
    void load_labels(const std::string& path);
    static uint32_t read_uint32_be(std::ifstream& f);
};

} // namespace data
} // namespace blade
