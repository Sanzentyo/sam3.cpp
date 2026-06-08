#pragma once

#include <algorithm>
#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace sam3::multiplex {

inline constexpr int padding_slot = -1;
inline constexpr int removed_slot = -1116;

struct State {
    std::vector<std::vector<int>> assignments;
    int allowed_bucket_capacity = 0;
    std::optional<std::vector<int>> object_ids;

    [[nodiscard]] int num_buckets() const { return static_cast<int>(assignments.size()); }

    [[nodiscard]] int multiplex_count() const {
        return assignments.empty() ? 0 : static_cast<int>(assignments.front().size());
    }

    [[nodiscard]] int total_valid_entries() const {
        int total = 0;
        for (const auto& bucket : assignments) {
            total +=
                static_cast<int>(std::ranges::count_if(bucket, [](int slot) { return slot >= 0; }));
        }
        return total;
    }

    [[nodiscard]] int total_non_padding_entries() const {
        int total = 0;
        for (const auto& bucket : assignments) {
            total += static_cast<int>(
                std::ranges::count_if(bucket, [](int slot) { return slot != padding_slot; }));
        }
        return total;
    }

    [[nodiscard]] int available_slots() const {
        return (num_buckets() * allowed_bucket_capacity) - total_non_padding_entries();
    }

    void validate() const {
        if (assignments.empty()) {
            throw std::invalid_argument("multiplex assignments must contain at least one bucket");
        }
        if (allowed_bucket_capacity <= 0) {
            throw std::invalid_argument("allowed_bucket_capacity must be positive");
        }
        const int mux_count = multiplex_count();
        if (allowed_bucket_capacity > mux_count) {
            throw std::invalid_argument("allowed_bucket_capacity cannot exceed multiplex_count");
        }

        std::vector<bool> seen(static_cast<size_t>(total_non_padding_entries()), false);
        for (const auto& bucket : assignments) {
            if (static_cast<int>(bucket.size()) != mux_count) {
                throw std::invalid_argument("all multiplex buckets must have equal size");
            }
            const int non_padding = static_cast<int>(
                std::ranges::count_if(bucket, [](int slot) { return slot != padding_slot; }));
            if (non_padding > allowed_bucket_capacity) {
                throw std::invalid_argument("bucket exceeds allowed capacity");
            }
            for (const int slot : bucket) {
                if (slot < 0) {
                    if (slot != padding_slot && slot != removed_slot) {
                        throw std::invalid_argument("unknown negative multiplex slot marker");
                    }
                    continue;
                }
                if (static_cast<size_t>(slot) >= seen.size()) {
                    throw std::invalid_argument("multiplex object index exceeds valid range");
                }
                if (seen[static_cast<size_t>(slot)]) {
                    throw std::invalid_argument("multiplex object indices must be unique");
                }
                seen[static_cast<size_t>(slot)] = true;
            }
        }
        if (object_ids && static_cast<int>(object_ids->size()) != total_valid_entries()) {
            throw std::invalid_argument("object_ids must map one-to-one to valid entries");
        }
    }

    [[nodiscard]] std::vector<float> mux(std::span<const float> data, int feature_count) const {
        validate_feature_contract(data, feature_count, total_valid_entries(), "mux");
        std::vector<float> out(
            static_cast<size_t>(num_buckets() * multiplex_count() * feature_count), 0.0f);
        for_each_valid_slot([&](int bucket_index, int slot_index, int object_index) {
            const size_t src = static_cast<size_t>(object_index * feature_count);
            const size_t dst = static_cast<size_t>(
                ((bucket_index * multiplex_count()) + slot_index) * feature_count);
            std::ranges::copy(data.subspan(src, static_cast<size_t>(feature_count)),
                              out.begin() + dst);
        });
        return out;
    }

    [[nodiscard]] std::vector<float> demux(std::span<const float> muxed, int feature_count) const {
        validate_feature_contract(muxed, feature_count, num_buckets() * multiplex_count(), "demux");
        std::vector<float> out(static_cast<size_t>(total_valid_entries() * feature_count), 0.0f);
        for_each_valid_slot([&](int bucket_index, int slot_index, int object_index) {
            const size_t src = static_cast<size_t>(
                ((bucket_index * multiplex_count()) + slot_index) * feature_count);
            const size_t dst = static_cast<size_t>(object_index * feature_count);
            std::ranges::copy(muxed.subspan(src, static_cast<size_t>(feature_count)),
                              out.begin() + dst);
        });
        return out;
    }

private:
    template <class Fn>
    void for_each_valid_slot(Fn&& fn) const {
        for (int bucket_index = 0; bucket_index < num_buckets(); ++bucket_index) {
            for (int slot_index = 0; slot_index < multiplex_count(); ++slot_index) {
                const int object_index =
                    assignments[static_cast<size_t>(bucket_index)][static_cast<size_t>(slot_index)];
                if (object_index >= 0) {
                    fn(bucket_index, slot_index, object_index);
                }
            }
        }
    }

    static void validate_feature_contract(std::span<const float> data,
                                          int feature_count,
                                          int rows,
                                          std::string_view op) {
        if (feature_count <= 0) {
            throw std::invalid_argument(std::format("{} feature_count must be positive", op));
        }
        const auto expected = static_cast<size_t>(rows * feature_count);
        if (data.size() != expected) {
            throw std::invalid_argument(
                std::format("{} expected {} floats, got {}", op, expected, data.size()));
        }
    }
};

[[nodiscard]] inline State make_sequential_state(
    int num_valid_entries,
    int multiplex_count,
    int eval_multiplex_count = -1,
    std::optional<std::vector<int>> object_ids = std::nullopt) {
    if (num_valid_entries <= 0) {
        throw std::invalid_argument("num_valid_entries must be positive");
    }
    if (multiplex_count <= 0) {
        throw std::invalid_argument("multiplex_count must be positive");
    }
    const int allowed_capacity = eval_multiplex_count < 0 ? multiplex_count : eval_multiplex_count;
    if (allowed_capacity <= 0 || allowed_capacity > multiplex_count) {
        throw std::invalid_argument("eval_multiplex_count must be in 1..multiplex_count");
    }
    const int num_buckets = (num_valid_entries + allowed_capacity - 1) / allowed_capacity;

    State state;
    state.allowed_bucket_capacity = allowed_capacity;
    state.object_ids = std::move(object_ids);
    state.assignments.reserve(static_cast<size_t>(num_buckets));

    int next_object = 0;
    for (int bucket_index = 0; bucket_index < num_buckets; ++bucket_index) {
        std::vector<int> bucket(static_cast<size_t>(multiplex_count), padding_slot);
        for (int slot_index = 0; slot_index < allowed_capacity && next_object < num_valid_entries;
             ++slot_index) {
            bucket[static_cast<size_t>(slot_index)] = next_object++;
        }
        state.assignments.push_back(std::move(bucket));
    }
    state.validate();
    return state;
}

}  // namespace sam3::multiplex
