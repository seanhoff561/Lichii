// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_SPIKE_LOG=<ms>: one log line for each presented frame that took at least <ms>, naming
// what happened during it, plus a frame time summary every 10 s. Costs nothing when unset.
namespace VideoCommon::Nx2Spike {

enum class Counter : u32 {
    Pipelines,     ///< Graphics and compute pipelines made during the frame
    PipelineNs,    ///< Time the GPU back thread spent starting them
    HoldNs,        ///< Time VulkanWorker waited for a pipeline still being built
    Uploads,       ///< Images refreshed from guest memory
    UploadNs,      ///< Time spent on them
    Images,        ///< Images created
    CpuReads,      ///< Guest CPU reads of GPU-written memory
    Finishes,      ///< Scheduler::Finish calls
    FinishNs,      ///< Time spent in them
    FrontIdleNs,   ///< Time the GPU front waited for the guest to submit something
    FrontWaitNs,   ///< Time the front waited for the back
    BackIdleNs,    ///< Time the back waited for the front
    BackHookNs,    ///< Of that, time in its idle hook (a deferred submit)
    FrontSleepNs,  ///< Of the front's wait, time asleep rather than spinning
    WakeLateNs,    ///< Time from the back's last wake to the front running again
    GpuWaitNs,     ///< Time threads waited for host GPU ticks
    SubmitNs,      ///< Time in queue submits
    AcquireFrameNs, ///< Time the back waited for a free presentation frame
    PresentNs,     ///< Time the present thread spent acquiring and presenting
    Count,
};

/// Totals logged with the frame time summary.
enum class Stat : u32 {
    AcquireChecks,     ///< Semaphore acquires
    AcquireEarly,      ///< Already met before waiting for the back
    AcquireStale,      ///< Met before the wait but not after it
    AcquireQueued,     ///< Not met, but a release queued before it meets it
    MacroReads,        ///< Macro DMA reads
    MacroGpuWritten,   ///< On pages GPU work wrote
    MacroWritersQueued, ///< With compute or copy work still queued
    MacroChanged,      ///< Words differed after waiting for the back
    MacroChangedGpu,   ///< Of those, on GPU-written pages
    Count,
};

inline std::atomic<u64> g_threshold_ms = [] {
    const char* const text = std::getenv("EDEN_NX2_SPIKE_LOG");
    return text ? static_cast<u64>(std::atoll(text)) : u64{0};
}();

inline std::array<std::atomic<u64>, static_cast<size_t>(Counter::Count)> g_counters{};

inline std::array<std::atomic<u64>, static_cast<size_t>(Stat::Count)> g_stats{};

inline void Note(Stat stat) {
    g_stats[static_cast<size_t>(stat)].fetch_add(1, std::memory_order_relaxed);
}

inline void Reload() {
    const char* const text = std::getenv("EDEN_NX2_SPIKE_LOG");
    g_threshold_ms.store(text ? static_cast<u64>(std::atoll(text)) : 0, std::memory_order_relaxed);
}

inline bool Enabled() {
    return g_threshold_ms.load(std::memory_order_relaxed) != 0;
}

inline void Add(Counter counter, u64 value = 1) {
    if (Enabled()) {
        g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
    }
}

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Times a span into a counter.
class Timer {
public:
    explicit Timer(Counter counter_) : counter{counter_}, start{Enabled() ? NowNs() : 0} {}
    ~Timer() {
        if (start != 0) {
            Add(counter, NowNs() - start);
        }
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    Counter counter;
    u64 start;
};

/// Called once per presented frame by the thread that presents.
void Frame();

} // namespace VideoCommon::Nx2Spike
