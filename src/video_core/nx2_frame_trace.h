// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>

namespace VideoCommon {

/// Process start, for EDEN_NX2_FRAME_TRACE.
inline const std::chrono::steady_clock::time_point nx2_frame_trace_epoch =
    std::chrono::steady_clock::now();

/// EDEN_NX2_FRAME_TRACE=<seconds>: for 100 ms from that long after start, log every draw, clear,
/// copy and blit on full-size targets in submission order (debug aid, one shared timeline).
inline bool Nx2FrameTraceActive() {
    static const double start = [] {
        const char* const value = std::getenv("EDEN_NX2_FRAME_TRACE");
        return value != nullptr ? std::strtod(value, nullptr) : -1.0;
    }();
    if (start < 0.0) {
        return false;
    }
    static std::atomic<unsigned> logged{};
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                     nx2_frame_trace_epoch)
                           .count();
    return now >= start && now < start + 0.1 && logged++ < 6000;
}

} // namespace VideoCommon
