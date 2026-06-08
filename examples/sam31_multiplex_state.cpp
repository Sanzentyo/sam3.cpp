#include "sam3_multiplex.h"
#include "sam3_sam31.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

struct Args {
    bool dump_assignments = false;
    int objects = 17;
    int multiplex_count = sam3::sam31::multiplex_count;
    int feature_count = 3;
};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string{message});
    }
}

void check_state_shape(int objects, int multiplex_count, int expected_buckets) {
    const auto state = sam3::multiplex::make_sequential_state(objects, multiplex_count);
    expect(state.num_buckets() == expected_buckets, "unexpected bucket count");
    expect(state.multiplex_count() == multiplex_count, "unexpected multiplex count");
    expect(state.total_valid_entries() == objects, "unexpected valid entry count");
    expect(state.available_slots() == expected_buckets * multiplex_count - objects,
           "unexpected available slot count");
}

void check_mux_roundtrip(int objects, int multiplex_count, int feature_count) {
    const auto state = sam3::multiplex::make_sequential_state(objects, multiplex_count);
    std::vector<float> data(static_cast<size_t>(objects * feature_count));
    std::iota(data.begin(), data.end(), 1.0f);

    const auto muxed = state.mux(std::span<const float>{data}, feature_count);
    const auto demuxed = state.demux(std::span<const float>{muxed}, feature_count);
    expect(demuxed == data, "demux(mux(x)) did not round-trip");

    const auto valid_values = static_cast<size_t>(objects * feature_count);
    const auto non_zero = static_cast<size_t>(
        std::ranges::count_if(muxed, [](float value) { return value != 0.0f; }));
    expect(non_zero == valid_values, "padding slots must stay zero after mux");
}

bool parse_int(std::string_view text, int& out) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value <= 0) {
        return false;
    }
    out = value;
    return true;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::format("missing value for {}", name));
            }
            return argv[++i];
        };
        if (arg == "--dump-assignments") {
            args.dump_assignments = true;
        } else if (arg == "--objects") {
            if (!parse_int(value(arg), args.objects)) {
                throw std::invalid_argument("invalid --objects");
            }
        } else if (arg == "--multiplex-count") {
            if (!parse_int(value(arg), args.multiplex_count)) {
                throw std::invalid_argument("invalid --multiplex-count");
            }
        } else if (arg == "--features") {
            if (!parse_int(value(arg), args.feature_count)) {
                throw std::invalid_argument("invalid --features");
            }
        } else {
            throw std::invalid_argument(std::format("unknown argument {}", arg));
        }
    }
    return args;
}

void print_float_array(std::span<const float> values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << std::format("{:.1f}", values[i]);
    }
    std::cout << "]";
}

void dump_assignments_json(const Args& args) {
    const auto state = sam3::multiplex::make_sequential_state(args.objects, args.multiplex_count);
    std::vector<float> data(static_cast<size_t>(args.objects * args.feature_count));
    std::iota(data.begin(), data.end(), 1.0f);
    const auto muxed = state.mux(std::span<const float>{data}, args.feature_count);
    const auto demuxed = state.demux(std::span<const float>{muxed}, args.feature_count);

    std::cout << "{";
    std::cout << std::format("\"objects\":{},", args.objects);
    std::cout << std::format("\"multiplex_count\":{},", args.multiplex_count);
    std::cout << std::format("\"feature_count\":{},", args.feature_count);
    std::cout << "\"assignments\":[";
    for (size_t b = 0; b < state.assignments.size(); ++b) {
        if (b != 0) {
            std::cout << ",";
        }
        std::cout << "[";
        for (size_t s = 0; s < state.assignments[b].size(); ++s) {
            if (s != 0) {
                std::cout << ",";
            }
            std::cout << state.assignments[b][s];
        }
        std::cout << "]";
    }
    std::cout << "],\"input\":";
    print_float_array(data);
    std::cout << ",\"muxed\":";
    print_float_array(muxed);
    std::cout << ",\"demuxed\":";
    print_float_array(demuxed);
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        if (args.dump_assignments) {
            dump_assignments_json(args);
            return 0;
        }
        constexpr int mux_count = sam3::sam31::multiplex_count;
        check_state_shape(1, mux_count, 1);
        check_state_shape(16, mux_count, 1);
        check_state_shape(17, mux_count, 2);
        check_state_shape(33, mux_count, 3);
        check_mux_roundtrip(17, mux_count, 3);
        check_mux_roundtrip(33, mux_count, 8);
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam31_multiplex_state failed: {}\n", ex.what());
        return 1;
    }

    std::cout << "sam31_multiplex_state ok\n";
    return 0;
}
