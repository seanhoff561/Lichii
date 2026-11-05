// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace Tegra {

template <typename Timing>
[[nodiscard]] u64 ReadGpuTimestamp(const Timing& timing, bool ampere, Settings::GpuClock clock) {
    if (ampere) {
        // Switch 2 timestamps are wall-clock nanoseconds when the emulator handles the report,
        // so a frame's GPU time reads as the emulator's whole frame time. Dynamic-resolution
        // titles then pick their lowest resolution (Ocarina of Time: a third of the output).
        // EDEN_NX2_GPU_TIME_SCALE=<n> makes time between reports n times shorter.
        static const u64 scale = [] {
            const char* value = std::getenv("EDEN_NX2_GPU_TIME_SCALE");
            const u64 parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 1;
            return parsed == 0 ? u64{1} : parsed;
        }();
        const u64 now = static_cast<u64>(timing.GetGlobalTimeNs().count());
        if (scale == 1) {
            return now;
        }
        static const u64 base = now;
        return base + (now - base) / scale;
    }
    const u64 ticks = timing.GetGPUTicks();
    switch (clock) {
    case Settings::GpuClock::Boost:
        return ticks / 256;
    case Settings::GpuClock::Overclock:
        return ticks / 512;
    default:
        return ticks;
    }
}

} // namespace Tegra
