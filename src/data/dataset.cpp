#include "blade/data/dataset.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace blade {
namespace data {

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset,
                       size_t batch_size, bool shuffle, size_t num_workers)
    : dataset_(std::move(dataset)),
      batch_size_(batch_size),
      shuffle_(shuffle),
      num_workers_(num_workers) {
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
    // TODO: stack inputs and labels along dim 0
    (void)samples;
    return {Tensor({1}), Tensor({1})};
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
