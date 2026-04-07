#include "blade/data/dataset.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace blade {
namespace data {

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset,
                       size_t batch_size, bool shuffle)
    : dataset_(std::move(dataset)),
      batch_size_(batch_size),
      shuffle_(shuffle) {
    indices_.resize(dataset_->size());
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_) reshuffle();
}

void DataLoader::reshuffle() {
    static std::mt19937 rng(std::random_device{}());
    std::shuffle(indices_.begin(), indices_.end(), rng);
}

size_t DataLoader::num_batches() const {
    return (dataset_->size() + batch_size_ - 1) / batch_size_;
}

Sample DataLoader::collate(const std::vector<Sample>& samples) const {
    if (samples.empty())
        throw std::runtime_error("collate: empty batch");
    const size_t n = samples.size();
    const auto& sample_shape = samples[0].first.shape();
    const size_t sample_numel = samples[0].first.numel();

    // Batch input shape: (n, *sample_shape)
    std::vector<size_t> batch_shape = {n};
    batch_shape.insert(batch_shape.end(), sample_shape.begin(), sample_shape.end());
    Tensor inputs(batch_shape);
    float* pi = inputs.data_ptr();
    for (size_t i = 0; i < n; ++i) {
        const float* ps = samples[i].first.data_ptr();
        std::copy(ps, ps + sample_numel, pi + i * sample_numel);
    }

    // Labels: (n,) — each label tensor holds one value
    Tensor labels({n});
    float* pl = labels.data_ptr();
    for (size_t i = 0; i < n; ++i)
        pl[i] = samples[i].second.data_ptr()[0];

    return {inputs, labels};
}

DataLoader::Iterator::Iterator(DataLoader* dl, size_t pos)
    : dl_(dl), pos_(pos) {}

Sample DataLoader::Iterator::operator*() const {
    size_t end = std::min(pos_ + dl_->batch_size_, dl_->dataset_->size());
    std::vector<Sample> batch;
    batch.reserve(end - pos_);
    for (size_t i = pos_; i < end; ++i)
        batch.push_back(dl_->dataset_->get(dl_->indices_[i]));
    return dl_->collate(batch);
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    pos_ += dl_->batch_size_;
    return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return pos_ < other.pos_;
}

bool DataLoader::Iterator::operator==(const Iterator& other) const {
    return pos_ >= other.pos_;
}

DataLoader::Iterator DataLoader::begin() {
    if (shuffle_) reshuffle();
    return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end() {
    return Iterator(this, dataset_->size());
}

} // namespace data
} // namespace blade
