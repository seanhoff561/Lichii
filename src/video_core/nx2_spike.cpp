// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#include "common/thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_spike.h"

namespace Dynarmic {
extern std::atomic<u64> g_nx2_jit_ns;
extern std::atomic<u64> g_nx2_jit_blocks;
extern std::atomic<u64> g_nx2_jit_flushes;
} // namespace Dynarmic

namespace VideoCommon::Nx2Spike {

namespace {

#ifdef _WIN32
// EDEN_NX2_WS_WATCH: who touches pages that are not in the working set, by thread, faulting
// code and address.
void WsWatch() {
    using Init = BOOL(WINAPI*)(HANDLE);
    using Get = BOOL(WINAPI*)(HANDLE, PPSAPI_WS_WATCH_INFORMATION_EX, PDWORD);
    const HMODULE kernel = GetModuleHandleA("kernel32.dll");
    const auto init = reinterpret_cast<Init>(GetProcAddress(kernel, "K32InitializeProcessForWsWatch"));
    const auto get = reinterpret_cast<Get>(GetProcAddress(kernel, "K32GetWsChangesEx"));
    if (!init || !get || !init(GetCurrentProcess())) {
        LOG_WARNING(HW_GPU, "No working set watch: {}", GetLastError());
        return;
    }
    const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
    std::vector<PSAPI_WS_WATCH_INFORMATION_EX> buffer(1 << 18);
    std::map<u64, u64> by_thread, by_pc, by_region;
    auto window = std::chrono::steady_clock::now();
    u64 total = 0;
    for (;;) {
        DWORD bytes = static_cast<DWORD>(buffer.size() * sizeof(buffer[0]));
        if (get(GetCurrentProcess(), buffer.data(), &bytes)) {
            for (const auto& entry : buffer) {
                if (entry.BasicInfo.FaultingPc == nullptr) {
                    break;
                }
                const u64 pc = reinterpret_cast<u64>(entry.BasicInfo.FaultingPc);
                ++total;
                ++by_thread[entry.FaultingThreadId];
                ++by_pc[pc >= exe && pc < exe + 0x10000000 ? pc - exe : pc];
                ++by_region[reinterpret_cast<u64>(entry.BasicInfo.FaultingVa) >> 24];
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - window >= std::chrono::seconds{10}) {
            window = now;
            const auto top = [](const std::map<u64, u64>& counts) {
                std::vector<std::pair<u64, u64>> sorted;
                for (const auto& [key, value] : counts) {
                    sorted.emplace_back(value, key);
                }
                std::sort(sorted.rbegin(), sorted.rend());
                std::string text;
                for (size_t i = 0; i < std::min<size_t>(sorted.size(), 14); ++i) {
                    text += fmt::format(" {:x}:{}", sorted[i].second, sorted[i].first);
                }
                return text;
            };
            std::string threads;
            for (const auto& [id, count] : by_thread) {
                std::wstring name;
                if (const HANDLE handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE,
                                                     static_cast<DWORD>(id))) {
                    PWSTR description = nullptr;
                    if (SUCCEEDED(GetThreadDescription(handle, &description)) && description) {
                        name = description;
                        LocalFree(description);
                    }
                    CloseHandle(handle);
                }
                std::string narrow;
                for (const wchar_t c : name) {
                    narrow += static_cast<char>(c);
                }
                threads += fmt::format(" {}/{}:{}", id, narrow, count);
            }
            LOG_INFO(HW_GPU, "NX2 ws faults {}; threads{}", total, threads);
            LOG_INFO(HW_GPU, "NX2 ws pc{}", top(by_pc));
            LOG_INFO(HW_GPU, "NX2 ws region{}", top(by_region));
            by_thread.clear();
            by_pc.clear();
            by_region.clear();
            total = 0;
        }
        Sleep(10);
    }
}
#endif

u32 PageFaults() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS info{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
        return info.PageFaultCount;
    }
#endif
    return 0;
}

} // Anonymous namespace

void Frame() {
    const u64 threshold_ms = g_threshold_ms.load(std::memory_order_relaxed);
    if (threshold_ms == 0) {
        return;
    }
    static u64 previous_ns = 0;
    static u64 window_start_ns = 0;
    static u32 previous_faults = 0;
    static u64 frame_number = 0;
    static std::vector<float> frames;
    static u64 previous_jit_ns = 0;
    static u64 previous_jit_blocks = 0;
    static u64 previous_jit_flushes = 0;
#ifdef _WIN32
    static const bool watch = [] {
        if (std::getenv("EDEN_NX2_WS_WATCH") != nullptr) {
            std::thread(WsWatch).detach();
        }
        return true;
    }();
    (void)watch;
#endif
    static const bool watchdog = [] {
        if (std::getenv("EDEN_NX2_SPIKE_WATCH") != nullptr) {
            std::thread([] {
                u64 logged_frame = 0;
                u32 snapshots = 0;
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    const u64 start = Common::g_nx2_frame_start_ns.load(std::memory_order_relaxed);
                    const u64 age = NowNs() - start;
                    if (age < 48'000'000) {
                        continue;
                    }
                    if (logged_frame != start) {
                        logged_frame = start;
                        snapshots = 0;
                    }
                    if (snapshots < 3) {
                        ++snapshots;
                        LOG_INFO(HW_GPU, "NX2 watch {} ms: {}", age / 1'000'000,
                                 VideoCommon::GpuSplit::Get().DebugState());
                        std::this_thread::sleep_for(std::chrono::milliseconds{4});
                    }
                }
            }).detach();
        }
        return true;
    }();
    (void)watchdog;
    const u64 now = NowNs();
    Common::g_nx2_frame_start_ns.store(now, std::memory_order_relaxed);
    const u64 jit_ns = Dynarmic::g_nx2_jit_ns.load(std::memory_order_relaxed);
    const u64 jit_blocks = Dynarmic::g_nx2_jit_blocks.load(std::memory_order_relaxed);
    const u64 jit_flushes = Dynarmic::g_nx2_jit_flushes.load(std::memory_order_relaxed);
    const u32 faults = PageFaults();
    std::array<u64, static_cast<size_t>(Counter::Count)> values;
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = g_counters[index].exchange(0, std::memory_order_relaxed);
    }
    ++frame_number;
    if (previous_ns != 0) {
        const float ms = static_cast<float>(now - previous_ns) / 1'000'000.0f;
        frames.push_back(ms);
        if (ms >= static_cast<float>(threshold_ms)) {
            const auto value = [&](Counter counter) {
                return values[static_cast<size_t>(counter)];
            };
            const auto millis = [&](Counter counter) {
                return static_cast<double>(value(counter)) / 1'000'000.0;
            };
            LOG_INFO(HW_GPU,
                     "NX2 spike frame {} {:.1f} ms: pipelines {} ({:.1f} ms back, {:.1f} ms held) "
                     "uploads {} ({:.1f} ms) images {} cpu reads {} finishes {} ({:.1f} ms) "
                     "page faults {} front idle {:.1f} ms jit {:.1f} ms {} blocks {} flushes "
                     "front wait {:.1f} (asleep {:.1f}, wake late {:.1f}) "
                     "back idle {:.1f} (hook {:.1f}) gpu wait {:.1f} submit {:.1f} "
                     "frame wait {:.1f} present {:.1f}",
                     frame_number, ms, value(Counter::Pipelines), millis(Counter::PipelineNs),
                     millis(Counter::HoldNs), value(Counter::Uploads), millis(Counter::UploadNs),
                     value(Counter::Images), value(Counter::CpuReads), value(Counter::Finishes),
                     millis(Counter::FinishNs), faults - previous_faults,
                     millis(Counter::FrontIdleNs),
                     static_cast<double>(jit_ns - previous_jit_ns) / 1'000'000.0,
                     jit_blocks - previous_jit_blocks, jit_flushes - previous_jit_flushes,
                     millis(Counter::FrontWaitNs), millis(Counter::FrontSleepNs),
                     millis(Counter::WakeLateNs), millis(Counter::BackIdleNs),
                     millis(Counter::BackHookNs), millis(Counter::GpuWaitNs),
                     millis(Counter::SubmitNs),
                     millis(Counter::AcquireFrameNs), millis(Counter::PresentNs));
        }
    }
    previous_ns = now;
    previous_faults = faults;
    previous_jit_ns = jit_ns;
    previous_jit_blocks = jit_blocks;
    previous_jit_flushes = jit_flushes;
    if (window_start_ns == 0) {
        window_start_ns = now;
    }
    if (now - window_start_ns >= 10'000'000'000ULL && !frames.empty()) {
        std::sort(frames.begin(), frames.end());
        const auto percentile = [&](size_t percent) {
            return frames[std::min(frames.size() - 1, frames.size() * percent / 100)];
        };
        const auto over = [&](float ms) {
            return static_cast<size_t>(frames.end() -
                                       std::lower_bound(frames.begin(), frames.end(), ms));
        };
        LOG_INFO(HW_GPU,
                 "NX2 frames {} in {:.1f} s: p50 {:.1f} p95 {:.1f} p99 {:.1f} worst {:.1f} ms; "
                 "over 20 ms {} over 33 ms {} over 50 ms {} over 100 ms {}",
                 frames.size(), static_cast<double>(now - window_start_ns) / 1e9, percentile(50),
                 percentile(95), percentile(99), frames.back(), over(20.0f), over(33.3f),
                 over(50.0f), over(100.0f));
        std::array<u64, static_cast<size_t>(Stat::Count)> stats;
        for (size_t index = 0; index < stats.size(); ++index) {
            stats[index] = g_stats[index].exchange(0, std::memory_order_relaxed);
        }
        LOG_INFO(HW_GPU,
                 "NX2 waits: acquires {} met early {} stale {} queued {}; macro reads {} on GPU "
                 "pages {} writers queued {} changed {} changed on GPU pages {}",
                 stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7],
                 stats[8]);
        frames.clear();
        window_start_ns = now;
    }
}

} // namespace VideoCommon::Nx2Spike
