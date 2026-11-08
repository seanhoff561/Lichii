// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>
#include <string>
#include <thread>

#include "common/adpf.h"
#include "common/error.h"
#include "common/event_wait.h"
#include "common/logging.h"
#include "common/assert.h"
#include "common/thread.h"
#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__HAIKU__)
#include <kernel/OS.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/string_util.h"
#include "common/windows/timer_resolution.h"
#if defined(_M_X64) || defined(__x86_64__)
#define NX2_SAMPLER 1
extern "C" IMAGE_DOS_HEADER __ImageBase;
#endif
#else
#if defined(__FreeBSD__)
#include <sys/cpuset.h>
#include <sys/_cpuset.h>
#include <pthread_np.h>
// Compatibility with CPUset
#define cpu_set_t cpuset_t
#elif defined(__DragonFly__) || defined(__OpenBSD__) || defined(__Bitrig__)
#include <pthread_np.h>
#endif
#include <pthread.h>
#include <sched.h>
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __linux__
#include <sys/resource.h>
#include <algorithm>

namespace {
constexpr int NICE_AUDIO = -16;
constexpr int NICE_URGENT_DISPLAY = -8;
constexpr int NICE_DISPLAY = -4;
constexpr int NICE_DEFAULT = 0;
constexpr int NICE_BACKGROUND = 10;

int LowestAllowedNice() {
    static const int lowest = [] {
        rlimit limit{};
        if (getrlimit(RLIMIT_NICE, &limit) != 0) {
            return 0;
        }
        if (limit.rlim_cur >= 40) {
            return -20;
        }
        return 20 - static_cast<int>(limit.rlim_cur);
    }();
    return lowest;
}

int NiceValueForPriority(Common::ThreadPriority priority) {
    const int wanted = [priority] {
        switch (priority) {
        case Common::ThreadPriority::Low: return NICE_BACKGROUND;
        case Common::ThreadPriority::Normal: return NICE_DEFAULT;
        case Common::ThreadPriority::High: return NICE_DISPLAY;
        case Common::ThreadPriority::VeryHigh: return NICE_URGENT_DISPLAY;
        case Common::ThreadPriority::Critical: return NICE_AUDIO;
        default: return NICE_DEFAULT;
        }
    }();
    return (std::max)(wanted, (std::min)(NICE_DEFAULT, LowestAllowedNice()));
}
} // Anonymous namespace
#endif

#ifdef __ANDROID__
#include <sys/utsname.h>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

namespace {
constexpr size_t ANDROID_MINIMUM_PERFORMANCE_CORES = 4;

// A core counts as a performance core while it is within this much of the fastest one.
constexpr s64 ANDROID_PERFORMANCE_CAPACITY_PERCENT = 50;

constexpr std::chrono::nanoseconds ANDROID_POLICY_POLL_INTERVAL = std::chrono::milliseconds{500};

enum class CoreGroup {
    Unrestricted,
    Performance,
    Efficiency,
};

struct ThreadPolicy {
    pid_t tid;
    CoreGroup group;
    s32 nice_value;
    bool has_nice;
};

struct CoreInfo {
    s64 weight;
    u64 midr;
    s32 cpu;
};

struct CpuTopologyState {
    std::mutex topology_mutex;
    cpu_set_t allowed{};
    cpu_set_t performance{};
    cpu_set_t efficiency{};
    bool separated = false;
    bool initialized = false;

    pid_t canary_tid = 0;
    cpu_set_t canary_mask{};
    bool canary_valid = false;

    std::atomic<s64> next_poll_ns{0};

    std::mutex policy_mutex;
    std::vector<ThreadPolicy> policies;
};

CpuTopologyState& State() {
    static CpuTopologyState* const state = new CpuTopologyState();
    return *state;
}

struct PolicyRegistration {
    ~PolicyRegistration() {
        const pid_t tid = gettid();
        ::Common::ADPF::RemoveCurrentThread();
        CpuTopologyState& state = State();
        std::scoped_lock lock{state.policy_mutex};
        std::erase_if(state.policies,
                      [tid](const ThreadPolicy& policy) { return policy.tid == tid; });
    }
};

thread_local PolicyRegistration t_policy_registration;

s32 PossibleCpuCount() {
    std::ifstream file("/sys/devices/system/cpu/possible");
    std::string list;
    if (file && std::getline(file, list) && !list.empty()) {
        s64 highest = -1;
        const char* cursor = list.c_str();
        while (*cursor != '\0') {
            char* end = nullptr;
            const s64 value = std::strtol(cursor, &end, 10);
            if (end == cursor) {
                break;
            }
            highest = (std::max)(highest, value);
            cursor = end;
            while (*cursor == '-' || *cursor == ',') {
                ++cursor;
            }
        }
        if (highest >= 0) {
            return static_cast<s32>((std::min<s64>)(highest + 1, CPU_SETSIZE));
        }
    }
    const s64 configured = sysconf(_SC_NPROCESSORS_CONF);
    if (configured > 0) {
        return static_cast<s32>((std::min<s64>)(configured, CPU_SETSIZE));
    }
    return static_cast<s32>((std::min<s64>)(std::thread::hardware_concurrency(), CPU_SETSIZE));
}

s64 ReadCpuScalar(s32 cpu, const char* node) {
    s64 value = 0;
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + node);
    if (!file || !(file >> value) || value <= 0) {
        return 0;
    }
    return value;
}

u64 ReadCpuMidr(s32 cpu) {
    u64 midr = 0;
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                       "/regs/identification/midr_el1");
    if (!file || !(file >> std::hex >> midr)) {
        return 0;
    }
    return midr;
}

std::vector<CoreInfo> CollectCores(const cpu_set_t& allowed, s32 total, const char* node,
                                   bool require_all) {
    std::vector<CoreInfo> cores;
    for (s32 cpu = 0; cpu < total; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const s64 weight = ReadCpuScalar(cpu, node);
        if (weight <= 0) {
            if (require_all) {
                return {};
            }
            LOG_WARNING(Common, "Could not read {} for CPU {}, treating it as an efficiency core",
                        node, cpu);
            continue;
        }
        cores.push_back(CoreInfo{weight, ReadCpuMidr(cpu), cpu});
    }
    return cores;
}

bool WeightsAreUniform(const std::vector<CoreInfo>& cores) {
    return std::all_of(cores.begin(), cores.end(),
                       [&](const CoreInfo& core) { return core.weight == cores.front().weight; });
}

bool MidrsAreDistinct(const std::vector<CoreInfo>& cores) {
    return std::none_of(cores.begin(), cores.end(),
                        [](const CoreInfo& core) { return core.midr == 0; }) &&
           std::any_of(cores.begin(), cores.end(),
                       [&](const CoreInfo& core) { return core.midr != cores.front().midr; });
}

void ComputeTopologyLocked(CpuTopologyState& state) {
    state.initialized = true;
    state.separated = false;
    CPU_ZERO(&state.allowed);
    CPU_ZERO(&state.performance);
    CPU_ZERO(&state.efficiency);

    if (sched_getaffinity(getpid(), sizeof(state.allowed), &state.allowed) != 0) {
        LOG_WARNING(Common, "Could not query process CPU affinity: {}",
                    ::Common::GetLastErrorMsg());
        return;
    }

    const s32 total = PossibleCpuCount();
    auto cores = CollectCores(state.allowed, total, "cpu_capacity", true);
    if (cores.empty() || (WeightsAreUniform(cores) && MidrsAreDistinct(cores))) {
        auto by_frequency = CollectCores(state.allowed, total, "cpufreq/cpuinfo_max_freq", false);
        if (!by_frequency.empty()) {
            cores = std::move(by_frequency);
        }
    }
    if (cores.empty()) {
        LOG_WARNING(Common, "Could not determine CPU topology, thread placement is disabled");
        return;
    }

    if (WeightsAreUniform(cores)) {
        if (MidrsAreDistinct(cores)) {
            LOG_WARNING(Common, "CPU clusters differ but rank identically, thread placement is "
                                "disabled");
        } else {
            LOG_INFO(Common, "CPU cores are symmetric, thread placement is disabled");
        }
        return;
    }

    std::sort(cores.begin(), cores.end(), [](const CoreInfo& lhs, const CoreInfo& rhs) {
        if (lhs.weight != rhs.weight) {
            return lhs.weight > rhs.weight;
        }
        return lhs.cpu < rhs.cpu;
    });

    const s64 fastest = cores.front().weight;
    size_t taken = 0;
    for (const auto& core : cores) {
        const bool fast_enough =
            core.weight * 100 >= fastest * ANDROID_PERFORMANCE_CAPACITY_PERCENT;
        if (!fast_enough && taken >= ANDROID_MINIMUM_PERFORMANCE_CORES) {
            break;
        }
        CPU_SET(core.cpu, &state.performance);
        ++taken;
    }
    if (taken == 0) {
        return;
    }

    for (s32 cpu = 0; cpu < total; ++cpu) {
        if (CPU_ISSET(cpu, &state.allowed) && !CPU_ISSET(cpu, &state.performance)) {
            CPU_SET(cpu, &state.efficiency);
        }
    }

    state.separated = CPU_COUNT(&state.efficiency) > 0;
    LOG_INFO(Common, "CPU topology: {} performance cores, {} efficiency cores, separation {}",
             CPU_COUNT(&state.performance), CPU_COUNT(&state.efficiency),
             state.separated ? "enabled" : "unavailable");
}

void EnsureTopologyLocked(CpuTopologyState& state) {
    if (!state.initialized) {
        ComputeTopologyLocked(state);
    }
}

void RefreshTopologyLocked(CpuTopologyState& state) {
    if (!state.initialized) {
        ComputeTopologyLocked(state);
        return;
    }
    cpu_set_t current;
    CPU_ZERO(&current);
    if (sched_getaffinity(getpid(), sizeof(current), &current) != 0) {
        return;
    }
    if (std::memcmp(&current, &state.allowed, sizeof(current)) != 0) {
        ComputeTopologyLocked(state);
    }
}

bool ApplyCoreGroupLocked(CpuTopologyState& state, pid_t tid, CoreGroup group,
                          bool* gone = nullptr) {
    const bool restrict_group = group != CoreGroup::Unrestricted && state.separated;
    const cpu_set_t* mask = &state.allowed;
    if (restrict_group) {
        mask = group == CoreGroup::Performance ? &state.performance : &state.efficiency;
    }
    if (CPU_COUNT(mask) == 0) {
        return false;
    }
    if (sched_setaffinity(tid, sizeof(*mask), mask) != 0) {
        if (gone != nullptr && errno == ESRCH) {
            *gone = true;
            return false;
        }
        LOG_WARNING(Common, "Could not restrict thread {} to its core group: {}", tid,
                    ::Common::GetLastErrorMsg());
        return false;
    }
    return true;
}

bool KernelPreservesRequestedAffinity() {
    utsname info{};
    if (uname(&info) != 0) {
        return false;
    }
    s32 major = 0;
    s32 minor = 0;
    if (std::sscanf(info.release, "%d.%d", &major, &minor) != 2) {
        return false;
    }
    return major > 6 || (major == 6 && minor >= 2);
}

void SnapshotCanaryLocked(CpuTopologyState& state) {
    state.canary_valid = false;
    if (!state.separated) {
        return;
    }
    for (const auto& policy : state.policies) {
        if (policy.group == CoreGroup::Unrestricted) {
            continue;
        }
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(policy.tid, sizeof(mask), &mask) != 0) {
            continue;
        }
        state.canary_tid = policy.tid;
        state.canary_mask = mask;
        state.canary_valid = true;
        return;
    }
}

bool DueForPoll(CpuTopologyState& state) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    s64 next = state.next_poll_ns.load(std::memory_order_relaxed);
    if (now_ns < next) {
        return false;
    }
    return state.next_poll_ns.compare_exchange_strong(
        next, now_ns + ANDROID_POLICY_POLL_INTERVAL.count(), std::memory_order_relaxed);
}

ThreadPolicy& AcquirePolicyLocked(CpuTopologyState& state, pid_t tid) {
    for (auto& policy : state.policies) {
        if (policy.tid == tid) {
            return policy;
        }
    }
    return state.policies.emplace_back(ThreadPolicy{tid, CoreGroup::Unrestricted, 0, false});
}

void SetCurrentThreadCoreGroup(CoreGroup group) {
    const pid_t tid = gettid();
    (void)&t_policy_registration;

    CoreGroup effective = group;
    if (tid == getpid() && group != CoreGroup::Unrestricted) {
        LOG_WARNING(Common, "Refusing to place the main thread: the CPU topology is read from it");
        effective = CoreGroup::Unrestricted;
    }

    CpuTopologyState& state = State();
    std::scoped_lock topology_lock{state.topology_mutex};
    EnsureTopologyLocked(state);
    ApplyCoreGroupLocked(state, tid, effective);

    std::scoped_lock policy_lock{state.policy_mutex};
    AcquirePolicyLocked(state, tid).group = effective;
    if (!state.canary_valid) {
        SnapshotCanaryLocked(state);
    }
}

void RememberCurrentThreadNice(pid_t tid, s32 nice_value) {
    (void)&t_policy_registration;
    CpuTopologyState& state = State();
    std::scoped_lock lock{state.policy_mutex};
    ThreadPolicy& policy = AcquirePolicyLocked(state, tid);
    policy.nice_value = nice_value;
    policy.has_nice = true;
}
} // Anonymous namespace
#endif

#include "common/cpu_features.h"
#ifdef ARCHITECTURE_x86_64
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "common/x64/rdtsc.h"
#endif

namespace Common {

void SetCurrentThreadPriority(ThreadPriority new_priority) {
#ifdef _WIN32
    int windows_priority = [&]() {
        switch (new_priority) {
        case ThreadPriority::Low: return THREAD_PRIORITY_BELOW_NORMAL;
        case ThreadPriority::Normal: return THREAD_PRIORITY_NORMAL;
        case ThreadPriority::High: return THREAD_PRIORITY_ABOVE_NORMAL;
        case ThreadPriority::VeryHigh: return THREAD_PRIORITY_HIGHEST;
        case ThreadPriority::Critical: return THREAD_PRIORITY_TIME_CRITICAL;
        default: return THREAD_PRIORITY_NORMAL;
        }
    }();
    SetThreadPriority(GetCurrentThread(), windows_priority);
#elif defined(__HAIKU__)
    // TODO: We have priorities for 3D rendering applications - may help lavapipe?
    int priority = [&]() {
        switch (new_priority) {
        case ThreadPriority::Low: return B_LOW_PRIORITY;
        case ThreadPriority::Normal: return B_NORMAL_PRIORITY;
        case ThreadPriority::High: return B_DISPLAY_PRIORITY;
        case ThreadPriority::VeryHigh: return B_URGENT_DISPLAY_PRIORITY;
        case ThreadPriority::Critical: return B_URGENT_PRIORITY;
        default: return B_NORMAL_PRIORITY;
        }
    }();
    set_thread_priority(find_thread(NULL), priority);
#elif defined(__ANDROID__)
    const int nice_value = NiceValueForPriority(new_priority);
    const pid_t tid = gettid();
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), nice_value) != 0) {
        LOG_WARNING(Common, "Could not set thread nice value to {}: {}", nice_value,
                    GetLastErrorMsg());
        return;
    }
    RememberCurrentThreadNice(tid, nice_value);
#elif defined(__linux__)
    const int nice_value = NiceValueForPriority(new_priority);
    if (setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
        LOG_DEBUG(Common, "Could not set thread nice value to {}: {}", nice_value,
                  GetLastErrorMsg());
    }
#else
    const s32 max_prio = sched_get_priority_max(SCHED_OTHER);
    const s32 min_prio = sched_get_priority_min(SCHED_OTHER);
    if (max_prio > min_prio) {
        const u32 level = (std::min)(static_cast<u32>(new_priority), 4U);
        sched_param params{};
        params.sched_priority =
            min_prio + static_cast<s32>(static_cast<u32>(max_prio - min_prio) * level) / 4;
        pthread_setschedparam(pthread_self(), SCHED_OTHER, &params);
    }
#endif
}

#ifdef NX2_SAMPLER
// EDEN_NX2_SAMPLE_PROFILE=1 (also from nx2_switches.txt): a sampling profiler for the named
// threads. About every millisecond each one is suspended for its instruction pointer and the
// return addresses near its stack pointer; every 10 s the hottest addresses per thread are logged
// as executable offsets ("ext" outside it), symbolized offline against the PDB. While a thread is
// suspended the sampler only makes system calls (no heap, no locks), so it cannot deadlock on a
// lock the thread holds.
namespace Nx2Sampler {
namespace {
struct SampledThread {
    HANDLE handle;
    std::string name;
};
std::mutex g_threads_mutex;
std::vector<SampledThread> g_threads;
std::once_flag g_started;

constexpr size_t StackWords = 96;
constexpr size_t CallerFrames = 8;

bool Enabled() {
    const char* value = std::getenv("EDEN_NX2_SAMPLE_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

constexpr size_t UnwindFrames = 32;

/// The executable's unwind entry for an address, looked up in its own exception directory: no
/// loader lock (the suspended thread could hold it).
const RUNTIME_FUNCTION* FindFunction(u64 base, const RUNTIME_FUNCTION* functions, size_t count,
                                     u64 address) {
    const u32 rva = static_cast<u32>(address - base);
    size_t low = 0;
    size_t high = count;
    while (low < high) {
        const size_t mid = (low + high) / 2;
        if (rva < functions[mid].BeginAddress) {
            high = mid;
        } else if (rva >= functions[mid].EndAddress) {
            low = mid + 1;
        } else {
            return &functions[mid];
        }
    }
    return nullptr;
}

/// Walks a suspended thread's executable frames with the x64 unwind data. Starting outside the
/// executable (driver, system), it resumes at the first stack word that returns into it. Returns
/// the number of frames (executable offsets) written; a fault reading a bad stack ends the walk.
size_t Unwind(CONTEXT context, u64 base, u64 end, const RUNTIME_FUNCTION* functions, size_t count,
              u64 stack_limit, u64* out) {
    size_t frames = 0;
    __try {
        // Through system libraries with their own unwind data (a blocked lock, a heap call).
        for (u32 hops = 0; hops < 16 && (context.Rip < base || context.Rip >= end); ++hops) {
            DWORD64 image_base{};
            const auto* const entry = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
            if (entry == nullptr) {
                break;
            }
            PVOID handler_data{};
            DWORD64 establisher{};
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip,
                             const_cast<PRUNTIME_FUNCTION>(entry), &context, &handler_data,
                             &establisher, nullptr);
            if (context.Rsp == 0 || context.Rsp >= stack_limit) {
                return 0;
            }
        }
        if (context.Rip < base || context.Rip >= end) {
            // Find a return address into the executable on the stack.
            bool found = false;
            for (u64 slot = context.Rsp; slot < context.Rsp + 0x800 && slot + 8 <= stack_limit;
                 slot += 8) {
                const u64 value = *reinterpret_cast<const u64*>(slot);
                if (value > base && value < end &&
                    FindFunction(base, functions, count, value) != nullptr) {
                    context.Rip = value;
                    context.Rsp = slot + 8;
                    found = true;
                    break;
                }
            }
            if (!found) {
                return 0;
            }
        }
        while (frames < UnwindFrames && context.Rip >= base && context.Rip < end) {
            out[frames++] = context.Rip - base;
            const RUNTIME_FUNCTION* const function =
                FindFunction(base, functions, count, context.Rip);
            if (function == nullptr) {
                // A leaf function: the return address is on top of the stack.
                context.Rip = *reinterpret_cast<const u64*>(context.Rsp);
                context.Rsp += 8;
                continue;
            }
            PVOID handler_data{};
            DWORD64 establisher{};
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, context.Rip,
                             const_cast<PRUNTIME_FUNCTION>(function), &context, &handler_data,
                             &establisher, nullptr);
            if (context.Rsp == 0 || context.Rsp >= stack_limit) {
                break;
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return frames;
}

void Run() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    const u64 base = reinterpret_cast<u64>(&__ImageBase);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(
        reinterpret_cast<const u8*>(&__ImageBase) + __ImageBase.e_lfanew);
    const u64 end = base + nt->OptionalHeader.SizeOfImage;
    const auto& exceptions =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
    const auto* const functions =
        reinterpret_cast<const RUNTIME_FUNCTION*>(base + exceptions.VirtualAddress);
    const size_t function_count = exceptions.Size / sizeof(RUNTIME_FUNCTION);
    struct Profile {
        u64 samples{};
        std::unordered_map<u64, u32> leaf;
        std::unordered_map<u64, u32> stack;
        std::unordered_map<u64, u32> edges; ///< (frame << 32) | its caller's frame
        /// Samples outside the executable: (module << 32) | the first executable frame.
        std::unordered_map<u64, u32> outside;
        /// The same samples by (module << 32) | offset in the module, in 16 byte steps.
        std::unordered_map<u64, u32> outside_rva;
    };
    std::map<std::string, Profile> profiles;
    std::vector<std::string> modules;
    const auto module_index = [&modules](u64 address) -> u64 {
        HMODULE owner{};
        char path[MAX_PATH]{};
        std::string name = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(address), &owner) &&
            GetModuleFileNameA(owner, path, MAX_PATH) != 0) {
            name = path;
            name = name.substr(name.find_last_of('\\') + 1);
        }
        const auto it = std::find(modules.begin(), modules.end(), name);
        if (it != modules.end()) {
            return static_cast<u64>(it - modules.begin());
        }
        modules.push_back(name);
        return modules.size() - 1;
    };
    std::vector<SampledThread> threads;
    auto last_report = std::chrono::steady_clock::now();
    u64 words[StackWords];
    const u64 spike_ns = [] {
        const char* const text = std::getenv("EDEN_NX2_SAMPLE_SPIKES");
        return text ? static_cast<u64>(std::atoll(text)) * 1'000'000 : u64{0};
    }();
    for (;;) {
        Sleep(1);
        if (spike_ns != 0) {
            const u64 now_ns = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            if (now_ns - g_nx2_frame_start_ns.load(std::memory_order_relaxed) < spike_ns) {
                continue;
            }
        }
        {
            std::scoped_lock lock{g_threads_mutex};
            threads = g_threads;
        }
        for (const SampledThread& thread : threads) {
            if (SuspendThread(thread.handle) == static_cast<DWORD>(-1)) {
                continue;
            }
            CONTEXT context{};
            context.ContextFlags = CONTEXT_FULL;
            const bool got = GetThreadContext(thread.handle, &context) != 0;
            u64 unwound[UnwindFrames];
            size_t unwound_count = 0;
            if (got) {
                // The stack's upper bound, from its committed region.
                MEMORY_BASIC_INFORMATION region{};
                u64 stack_limit = context.Rsp + 0x100000;
                if (VirtualQuery(reinterpret_cast<void*>(context.Rsp), &region, sizeof(region))) {
                    stack_limit = reinterpret_cast<u64>(region.BaseAddress) + region.RegionSize;
                }
                unwound_count = Unwind(context, base, end, functions, function_count,
                                       stack_limit, unwound);
            }
            ResumeThread(thread.handle);
            (void)words;
            SIZE_T read = 0;
            if (!got) {
                continue;
            }
            Profile& profile = profiles[thread.name];
            ++profile.samples;
            const u64 rip = context.Rip;
            ++profile.leaf[rip >= base && rip < end ? rip - base : 1];
            if ((rip < base || rip >= end) && unwound_count != 0) {
                ++profile.outside[(module_index(rip) << 32) | unwound[0]];
            }
            if (rip < base || rip >= end) {
                HMODULE owner{};
                if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       reinterpret_cast<LPCSTR>(rip), &owner)) {
                    const u64 rva = (rip - reinterpret_cast<u64>(owner)) & ~u64{15};
                    ++profile.outside_rva[(module_index(rip) << 32) | rva];
                }
            }
            // Inclusive counts: each frame of the unwound stack once (recursion counted once).
            (void)read;
            for (size_t i = 0; i < unwound_count; ++i) {
                bool duplicate = false;
                for (size_t j = 0; j < i; ++j) {
                    duplicate |= unwound[j] == unwound[i];
                }
                if (!duplicate) {
                    ++profile.stack[unwound[i]];
                }
                if (i + 1 < unwound_count) {
                    ++profile.edges[(unwound[i] << 32) | unwound[i + 1]];
                }
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report < std::chrono::seconds{10}) {
            continue;
        }
        last_report = now;
        for (auto& [name, profile] : profiles) {
            if (profile.samples < 100) {
                continue;
            }
            const auto top = [&](const std::unordered_map<u64, u32>& counts, size_t limit = 150) {
                std::vector<std::pair<u32, u64>> sorted;
                sorted.reserve(counts.size());
                for (const auto& [offset, count] : counts) {
                    sorted.emplace_back(count, offset);
                }
                std::sort(sorted.rbegin(), sorted.rend());
                std::string out;
                for (size_t i = 0; i < sorted.size() && i < limit; ++i) {
                    out += fmt::format(" {:#x}:{:.1f}", sorted[i].second,
                                       100.0 * sorted[i].first / profile.samples);
                }
                return out;
            };
            LOG_INFO(Common, "Sample profile {} ({} samples) leaf:{}", name, profile.samples,
                     top(profile.leaf));
            LOG_INFO(Common, "Sample profile {} ({} samples) stack:{}", name, profile.samples,
                     top(profile.stack));
            LOG_INFO(Common, "Sample profile {} ({} samples) edges:{}", name, profile.samples,
                     top(profile.edges, 400));
            std::string outside;
            for (const auto& [key, count] : profile.outside) {
                if (100.0 * count / profile.samples >= 0.1) {
                    outside += fmt::format(" {}@{:#x}:{:.1f}", modules[key >> 32],
                                           key & 0xffffffff, 100.0 * count / profile.samples);
                }
            }
            LOG_INFO(Common, "Sample profile {} ({} samples) outside:{}", name, profile.samples,
                     outside);
            std::string rvas;
            for (const auto& [key, count] : profile.outside_rva) {
                if (100.0 * count / profile.samples >= 0.1) {
                    rvas += fmt::format(" {}+{:#x}:{:.1f}", modules[key >> 32], key & 0xffffffff,
                                        100.0 * count / profile.samples);
                }
            }
            LOG_INFO(Common, "Sample profile {} ({} samples) rva:{}", name, profile.samples, rvas);
        }
        profiles.clear();
    }
}
} // Anonymous namespace

void Register(const char* name) {
    if (!Enabled()) {
        return;
    }
    HANDLE handle{};
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                         FALSE, 0)) {
        return;
    }
    {
        std::scoped_lock lock{g_threads_mutex};
        g_threads.push_back({handle, name});
    }
    std::call_once(g_started, [] { std::thread{Run}.detach(); });
}
} // namespace Nx2Sampler
#endif

void SetCurrentThreadName(const char* name) {
#ifdef NX2_SAMPLER
    Nx2Sampler::Register(name);
#endif
#ifdef _MSC_VER
    // Sets the debugger-visible name of the current thread.
    if (auto pf = (decltype(&SetThreadDescription))(void*)GetProcAddress(GetModuleHandle(TEXT("KernelBase.dll")), "SetThreadDescription"); pf)
        pf(GetCurrentThread(), UTF8ToUTF16W(name).data()); // Windows 10+
    else
        ; // No-op
#elif  defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__HAIKU__)
    rename_thread(find_thread(NULL), name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__) || defined(__CYGWIN__) || defined(__sun__) || defined(__glibc__) || defined(__managarm__)
    int ret = pthread_setname_np(pthread_self(), name);
    if (ret == ERANGE) {
        // Linux limits thread names to 15 characters and will outright reject any
        // attempt to set a longer name with ERANGE.
        char buf[16];
        size_t const len = std::min<size_t>(std::strlen(name), sizeof(buf) - 1);
        std::memcpy(buf, name, len);
        buf[len] = '\0';
        pthread_setname_np(pthread_self(), buf);
    }
#elif defined(_WIN32)
    // MinGW with the POSIX threading model does not support pthread_setname_np
    // See for reference
    // https://gitlab.freedesktop.org/mesa/mesa/-/blame/main/src/util/u_thread.c?ref_type=heads#L75
    (void)name;
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void ReserveCoreForCurrentThread() {
#ifdef _WIN32
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_PIN");
        return value != nullptr && value[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static std::vector<u32> reserved_cores;
    std::scoped_lock lock{mutex};
    ULONG length = 0;
    GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
    std::vector<u8> buffer(length);
    if (!GetSystemCpuSetInformation(
            reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), length, &length,
            GetCurrentProcess(), 0)) {
        return;
    }
    struct Set {
        ULONG id;
        u32 core;
    };
    std::vector<Set> sets;
    std::vector<u32> cores;
    for (ULONG offset = 0; offset < length;) {
        const auto* const info =
            reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (info->Type == CpuSetInformation) {
            sets.push_back({info->CpuSet.Id, info->CpuSet.CoreIndex});
            if (std::find(cores.begin(), cores.end(), info->CpuSet.CoreIndex) == cores.end()) {
                cores.push_back(info->CpuSet.CoreIndex);
            }
        }
        offset += info->Size;
    }
    if (cores.size() < 8) {
        return;
    }
    // From the second core up: the first takes most of the system's interrupts.
    const u32 core = cores[1 + reserved_cores.size()];
    reserved_cores.push_back(core);
    std::vector<ULONG> others;
    ULONG own = 0;
    for (const Set& set : sets) {
        if (set.core == core && own == 0) {
            own = set.id;
        }
        if (std::find(reserved_cores.begin(), reserved_cores.end(), set.core) ==
            reserved_cores.end()) {
            others.push_back(set.id);
        }
    }
    SetThreadSelectedCpuSets(GetCurrentThread(), &own, 1);
    SetProcessDefaultCpuSets(GetCurrentProcess(), others.data(),
                             static_cast<ULONG>(others.size()));
    LOG_INFO(Common, "Reserved core {} for this thread, {} sets for the rest", core, others.size());
#endif
}

void SetCurrentThreadToPerformanceCores() {
#if defined(__ANDROID__)
    if (ADPF::AddCurrentThread(ADPF::Session::Render)) {
        SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
        return;
    }
    SetCurrentThreadCoreGroup(CoreGroup::Performance);
#endif
}

void SetCurrentThreadToEfficiencyCores() {
#if defined(__ANDROID__)
    if (ADPF::AddCurrentThread(ADPF::Session::Background)) {
        SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
        return;
    }
    SetCurrentThreadCoreGroup(CoreGroup::Efficiency);
#endif
}

void SetCurrentThreadToBackgroundWork() {
#if defined(__ANDROID__)
    ADPF::AddCurrentThread(ADPF::Session::Background);
    SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
#endif
}

void SetCurrentThreadToAllCores() {
#if defined(__ANDROID__)
    ADPF::RemoveCurrentThread();
    SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
#endif
}

void RefreshThreadPolicies() {
#if defined(__ANDROID__)
    CpuTopologyState& state = State();
    std::scoped_lock topology_lock{state.topology_mutex};
    RefreshTopologyLocked(state);

    std::scoped_lock policy_lock{state.policy_mutex};
    std::erase_if(state.policies, [&state](const ThreadPolicy& policy) {
        bool gone = false;
        if (policy.has_nice &&
            setpriority(PRIO_PROCESS, static_cast<id_t>(policy.tid), policy.nice_value) != 0 &&
            errno == ESRCH) {
            gone = true;
        }
        if (!gone) {
            ApplyCoreGroupLocked(state, policy.tid, policy.group, &gone);
        }
        return gone;
    });
    SnapshotCanaryLocked(state);
#endif
}

void PollThreadPolicies() {
#if defined(__ANDROID__)
    static const bool needed = !KernelPreservesRequestedAffinity();
    if (!needed) {
        return;
    }

    CpuTopologyState& state = State();
    if (!DueForPoll(state)) {
        return;
    }

    pid_t tid;
    cpu_set_t expected;
    {
        std::scoped_lock lock{state.topology_mutex};
        if (!state.canary_valid) {
            return;
        }
        tid = state.canary_tid;
        expected = state.canary_mask;
    }

    cpu_set_t current;
    CPU_ZERO(&current);
    if (sched_getaffinity(tid, sizeof(current), &current) == 0 &&
        std::memcmp(&current, &expected, sizeof(current)) == 0) {
        return;
    }
    RefreshThreadPolicies();
#endif
}

#ifdef ARCHITECTURE_x86_64
// On Linux and UNIX systems, a futex would nominally be used to cover the costs
// the idea is that it's intuitivelly cheaper to use a direct instruction as opposed to a full futex call
// the underlying libc++ implementation uses pthread_cond_timedwait which MAY invoke a futex
// Let's pretend the OS is too expensive to jump into, and avoid ANY context switches
// this should *IN THEORY* lower CPU usage while just waiting for stuff effectively
// For windows the minimal quanta resolution is about 500us, and normal CRT cond var is 1.5ms(?)
// so may as well avoid that too
// Let's just give ALL platforms the same mechanisms (almost) for when they have umonitor OR waitpkg
#ifdef __clang__
__attribute__((target("waitpkg,mwaitx")))
#elif defined(__GNUC__)
#pragma GCC target("waitpkg")
#pragma GCC target("mwaitx")
#endif
bool Event::WaitFor(const std::chrono::nanoseconds time) {
#ifdef _WIN32
    auto const& caps = Common::g_cpu_caps;
    const auto deadline = EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()), time);
    const auto remaining = [deadline] {
        return deadline - std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch());
    };
    const u64 ticks_per_second = Common::g_wall_clock.NsToTicks(std::chrono::seconds{1});
    if (caps.monitorx) {
        while (true) {
            // Armed monitor, as per manual, MWAITX must be conditional if the condition isn't satisfied
            // to prevent a race condition.
            _mm_monitorx(reinterpret_cast<u64*>(std::addressof(is_set)), 0, 0);
            if (!is_set.load()) {
                const u32 cycles = MwaitxTimeoutCycles(remaining(), ticks_per_second);
                if (cycles != 0) {
                    // See here: https://github.com/torvalds/linux/blob/948a64995aca6820abefd17f1a4258f5835c5ad9/arch/x86/lib/delay.c#L93
                    // MWAITX accepts a 32-bit input timer which determines the total number of cycles to wait for
                    // NOT THE TOTAL ABSOLUTE TSC VALUE, it's just a delta
                    // BIT[1] = use a timer
                    // Hint = 0: Use C1 state when sleepy (means slower wakeup but better savings)
                    _mm_mwaitx(1 << 1, 0u, cycles);
                    continue;
                } else
                    return false; //timeout
            }
            bool expected = true;
            if (is_set.compare_exchange_weak(expected, false, std::memory_order_release))
                return true;
        }
    }
    if (caps.waitpkg) {
        // #UD If CPUID.7.0:ECX.WAITPKG[bit 5]=0.
        while (true) {
            _umonitor(std::addressof(is_set));
            if (!is_set.load()) {
                // UMWAIT also wakes early on interrupts, and its result (CF) only reports the OS
                // limit (IA32_UMWAIT_CONTROL). Wait again until the deadline, as for MWAITX. The
                // bounded budget keeps the absolute TSC deadline from wrapping.
                const u32 cycles = MwaitxTimeoutCycles(remaining(), ticks_per_second);
                if (cycles == 0)
                    return false; // timeout
                _umwait(0, Common::X64::FencedRDTSC() + cycles); // umwait is absolute time!!!
                continue;
            }
            bool expected = true;
            if (is_set.compare_exchange_weak(expected, false, std::memory_order_release))
                return true;
        }
    }
    while (!is_set.load() && remaining() > std::chrono::nanoseconds::zero())
        Common::Windows::SleepForOneTick();
    bool expected = true;
    return is_set.compare_exchange_strong(expected, false, std::memory_order_release);
#else
    std::unique_lock lk{mutex};
    const auto now = std::chrono::steady_clock::now();
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()), time)};
    if (!condvar.wait_until(lk, deadline, [this] { return is_set.load(); }))
        return false;
    is_set = false;
    return true;
#endif
}
#else
bool Event::WaitFor(const std::chrono::nanoseconds time) {
#ifdef _WIN32
    auto const end = Common::g_wall_clock.GetTimeNS() + time;
    while (!is_set.load() && end > Common::g_wall_clock.GetTimeNS())
        Common::Windows::SleepForOneTick();
    if (is_set.load())
        Reset();
    return true;
#else
    std::unique_lock lk{mutex};
    const auto now = std::chrono::steady_clock::now();
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()), time)};
    if (!condvar.wait_until(lk, deadline, [this] { return is_set.load(); }))
        return false;
    is_set = false;
    return true;
#endif
}
#endif

} // namespace Common
