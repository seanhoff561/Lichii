// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class ImageView;
class MemoryAllocator;
class Scheduler;

/// EDEN_NX2_FRAME_DUMP=<dir> reads back one whole frame: the color targets of every pass, the
/// images written by every compute dispatch and the presented image. EDEN_NX2_FRAME_DUMP_AT picks
/// the frame, in seconds after boot ("45") or as a frame number ("2700f").
class FrameDump {
public:
    explicit FrameDump(MemoryAllocator& memory_allocator, Scheduler& scheduler);
    ~FrameDump();

    [[nodiscard]] bool Capturing() const noexcept {
        return capturing;
    }

    /// Starts a pass when the framebuffer changes and captures the previous one.
    /// Returns true when copies were recorded, which ends the current render pass.
    bool BeginPass(const void* framebuffer, std::span<ImageView* const> colors);
    /// pixel_program: the draw's native pixel shader address (0 when none), matched to
    /// EDEN_NX2_SHADER_DUMP's file names.
    void NoteDraw(std::span<const u64> shader_hashes, u64 pixel_program = 0);
    /// Captures the current pass; returns true when copies were recorded.
    bool EndPass();
    void CaptureDispatch(u64 shader_hash, std::span<ImageView* const> written);
    void CapturePresent(ImageView& view);
    /// Waits for pending readbacks and writes them once they pass the staging budget.
    void DrainIfFull();
    void TickFrame();

private:
    struct Target {
        u32 slot;
        VkImage image;
        VideoCore::Surface::PixelFormat format;
        u32 width;
        u32 height;
        s32 level;
        s32 layer;
        GPUVAddr gpu_addr;
        bool rescaled;
        bool msaa;
    };

    struct Readback {
        vk::Buffer buffer;
        Target target;
        std::string name;
        std::string tags;
    };

    static std::optional<Target> MakeTarget(ImageView& view, u32 slot);
    bool Record(const Target& target, std::string name, std::string tags);
    void Drain();
    void Write(const Readback& readback, std::span<const u8> data);
    void WriteFile(const std::string& name, std::span<const u8> data);

    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    bool enabled{};
    bool capturing{};
    bool done{};
    bool truncated{};
    std::filesystem::path directory;
    std::filesystem::path trigger_file;
    u64 trigger_frame{~0ULL};
    double trigger_seconds{};
    std::chrono::steady_clock::time_point start;
    u64 frame{};

    std::ofstream manifest;
    std::vector<Readback> readbacks;
    size_t pending_bytes{};
    size_t written_bytes{};
    u32 sequence{};
    u32 images{};

    const void* pass_framebuffer{};
    std::vector<Target> pass_targets;
    std::vector<u64> pass_hashes;
    std::vector<u64> pass_programs;
    u32 pass_index{};
    u32 pass_first_draw{};
    u32 draws{};
    u32 dispatches{};
};

} // namespace Vulkan
