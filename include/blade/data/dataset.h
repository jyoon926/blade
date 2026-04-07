#pragma once
#include "blade/tensor.h"
#include <cstddef>
#include <vector>
#include <utility>
#include <memory>

namespace blade {
namespace data {

using Sample = std::pair<Tensor, Tensor>;  // (input, label)

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual Sample get(size_t idx) const = 0;
};

// Wraps a Dataset and yields mini-batches of (inputs, labels) tensors.
// Stacks individual samples along a new batch dimension (dim 0).
class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset,
               size_t batch_size,
               bool shuffle = true);

    struct Iterator {
        Iterator(DataLoader* dl, size_t pos);
        Sample operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
        bool operator==(const Iterator& other) const;
    private:
        DataLoader* dl_;
        size_t pos_;
    };

    Iterator begin();
    Iterator end();
    size_t num_batches() const;

private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;  // (possibly shuffled) sample order

    void reshuffle();
    Sample collate(const std::vector<Sample>& samples) const;
};

} // namespace data
} // namespace blade
