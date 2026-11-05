// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/settings.h"
#include "common/stb.h"
#include "common/zstd_compression.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

using VideoCore::Surface::PixelFormat;

constexpr size_t SIZE_CAP = size_t{500} << 20;
constexpr size_t STAGING_BUDGET = size_t{256} << 20;

template <typename T>
T Load(const u8* pixel, int index) {
    T value;
    std::memcpy(&value, pixel + index * sizeof(T), sizeof(T));
    return value;
}

template <typename T, typename Convert>
int Channels(const u8* pixel, int count, std::array<float, 4>& out, Convert&& convert) {
    for (int channel = 0; channel < count; ++channel) {
        out[channel] = convert(Load<T>(pixel, channel));
    }
    return count;
}

/// Unsigned float with a 5-bit exponent, as in half, B10G11R11 and friends.
float PackedFloat(u32 bits, int mantissa_bits) {
    const u32 exponent = bits >> mantissa_bits;
    const u32 mantissa = bits & ((1U << mantissa_bits) - 1);
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -14 - mantissa_bits);
    }
    if (exponent == 31) {
        return mantissa != 0 ? std::numeric_limits<float>::quiet_NaN()
                             : std::numeric_limits<float>::infinity();
    }
    return std::ldexp(static_cast<float>(mantissa | (1U << mantissa_bits)),
                      static_cast<int>(exponent) - 15 - mantissa_bits);
}

float Half(u16 bits) {
    const float magnitude = PackedFloat(bits & 0x7fff, 10);
    return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}

float Unorm8(u8 value) {
    return value / 255.0f;
}

float Unorm16(u16 value) {
    return value / 65535.0f;
}

float Identity(float value) {
    return value;
}

/// Returns the number of channels, or 0 when the format has no preview.
int DecodePixel(PixelFormat format, const u8* pixel, std::array<float, 4>& out) {
    out = {0.0f, 0.0f, 0.0f, 1.0f};
    switch (format) {
    case PixelFormat::A8B8G8R8_UNORM:
    case PixelFormat::A8B8G8R8_SRGB:
        return Channels<u8>(pixel, 4, out, Unorm8);
    case PixelFormat::B8G8R8A8_UNORM:
    case PixelFormat::B8G8R8A8_SRGB:
        out = {Unorm8(pixel[2]), Unorm8(pixel[1]), Unorm8(pixel[0]), Unorm8(pixel[3])};
        return 4;
    case PixelFormat::R8_UNORM:
        return Channels<u8>(pixel, 1, out, Unorm8);
    case PixelFormat::R8G8_UNORM:
        return Channels<u8>(pixel, 2, out, Unorm8);
    case PixelFormat::R16_UNORM:
        return Channels<u16>(pixel, 1, out, Unorm16);
    case PixelFormat::R16G16_UNORM:
        return Channels<u16>(pixel, 2, out, Unorm16);
    case PixelFormat::R16G16B16A16_UNORM:
        return Channels<u16>(pixel, 4, out, Unorm16);
    case PixelFormat::R16_FLOAT:
        return Channels<u16>(pixel, 1, out, Half);
    case PixelFormat::R16G16_FLOAT:
        return Channels<u16>(pixel, 2, out, Half);
    case PixelFormat::R16G16B16A16_FLOAT:
        return Channels<u16>(pixel, 4, out, Half);
    case PixelFormat::R32_FLOAT:
        return Channels<float>(pixel, 1, out, Identity);
    case PixelFormat::R32G32_FLOAT:
        return Channels<float>(pixel, 2, out, Identity);
    case PixelFormat::R32G32B32A32_FLOAT:
        return Channels<float>(pixel, 4, out, Identity);
    case PixelFormat::B10G11R11_FLOAT: {
        const u32 value = Load<u32>(pixel, 0);
        out = {PackedFloat(value & 0x7ff, 6), PackedFloat((value >> 11) & 0x7ff, 6),
               PackedFloat(value >> 22, 5), 1.0f};
        return 3;
    }
    case PixelFormat::A2B10G10R10_UNORM: {
        const u32 value = Load<u32>(pixel, 0);
        out = {(value & 0x3ff) / 1023.0f, ((value >> 10) & 0x3ff) / 1023.0f,
               ((value >> 20) & 0x3ff) / 1023.0f, (value >> 30) / 3.0f};
        return 4;
    }
    default:
        return 0;
    }
}

bool IsFloat(PixelFormat format) {
    switch (format) {
    case PixelFormat::R16_FLOAT:
    case PixelFormat::R16G16_FLOAT:
    case PixelFormat::R16G16B16A16_FLOAT:
    case PixelFormat::R32_FLOAT:
    case PixelFormat::R32G32_FLOAT:
    case PixelFormat::R32G32B32A32_FLOAT:
    case PixelFormat::B10G11R11_FLOAT:
        return true;
    default:
        return false;
    }
}

/// Float targets hold linear values, so the preview encodes them to sRGB; UNORM bytes are kept.
u8 PreviewByte(float value, bool encode) {
    if (!(value > 0.0f)) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    if (encode) {
        value = value <= 0.0031308f ? value * 12.92f
                                    : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
    }
    return static_cast<u8>(value * 255.0f + 0.5f);
}

std::string JsonNumber(double value) {
    return std::isfinite(value) ? fmt::format("{:.6g}", value) : "null";
}

template <typename T>
std::string JsonArray(std::span<T> values) {
    std::string out{"["};
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        if constexpr (std::is_floating_point_v<T>) {
            out += JsonNumber(values[i]);
        } else {
            out += fmt::format("{}", values[i]);
        }
    }
    return out + "]";
}

} // Anonymous namespace

FrameDump::FrameDump(MemoryAllocator& memory_allocator_, Scheduler& scheduler_)
    : memory_allocator{memory_allocator_}, scheduler{scheduler_} {
    const char* const dir = std::getenv("EDEN_NX2_FRAME_DUMP");
    if (!dir || *dir == '\0') {
        return;
    }
    enabled = true;
    directory = dir;
    start = std::chrono::steady_clock::now();
    const char* const at = std::getenv("EDEN_NX2_FRAME_DUMP_AT");
    const std::string_view value = at ? at : "60";
    if (value == "file") {
        // Captures once <dir>/capture.now exists, so a frame can be picked while playing.
        trigger_file = directory / "capture.now";
        trigger_seconds = std::numeric_limits<double>::infinity();
    } else if (value.ends_with('f')) {
        trigger_frame = std::strtoull(value.data(), nullptr, 10);
        trigger_seconds = std::numeric_limits<double>::infinity();
    } else {
        trigger_seconds = std::strtod(value.data(), nullptr);
    }
    LOG_INFO(Render_Vulkan, "Frame dump armed: dir={} at={}", directory.string(), value);
}

FrameDump::~FrameDump() = default;

std::optional<FrameDump::Target> FrameDump::MakeTarget(ImageView& view, u32 slot) {
    using namespace VideoCore::Surface;
    const Image* const image = view.BackingImage();
    if (!image) {
        return std::nullopt;
    }
    const auto& info = image->info;
    if (info.type == VideoCommon::ImageType::Buffer ||
        GetFormatType(info.format) != SurfaceType::ColorTexture ||
        DefaultBlockWidth(info.format) != 1 || DefaultBlockHeight(info.format) != 1) {
        return std::nullopt;
    }
    const bool rescaled = image->IsRescaled();
    const auto& resolution = Settings::values.resolution_info;
    const u32 width = rescaled ? resolution.ScaleUp(info.size.width) : info.size.width;
    const u32 height = rescaled ? resolution.ScaleUp(info.size.height) : info.size.height;
    const u32 level = static_cast<u32>(view.range.base.level);
    const bool volume = info.type == VideoCommon::ImageType::e3D;
    return Target{
        .slot = slot,
        .image = image->Handle(),
        .format = BytesPerBlock(view.format) == BytesPerBlock(info.format) ? view.format
                                                                           : info.format,
        .width = std::max(1U, width >> level),
        .height = std::max(1U, height >> level),
        .depth = volume ? std::max(1U, info.size.depth >> level) : 1U,
        .level = view.range.base.level,
        .layer = volume ? 0 : view.range.base.layer,
        .gpu_addr = image->gpu_addr,
        .rescaled = rescaled,
        .msaa = info.num_samples > 1,
    };
}

bool FrameDump::BeginPass(const void* framebuffer, std::span<ImageView* const> colors) {
    if (framebuffer == pass_framebuffer) {
        return false;
    }
    const bool recorded = EndPass();
    pass_framebuffer = framebuffer;
    pass_targets.clear();
    for (u32 slot = 0; slot < colors.size(); ++slot) {
        if (!colors[slot]) {
            continue;
        }
        if (const auto target = MakeTarget(*colors[slot], slot)) {
            pass_targets.push_back(*target);
        }
    }
    pass_hashes.clear();
    pass_programs.clear();
    pass_constants.clear();
    pass_constants_set = false;
    pass_first_draw = draws;
    return recorded;
}

void FrameDump::NoteDraw(std::span<const u64> shader_hashes, u64 pixel_program) {
    ++draws;
    if (pixel_program != 0 && pass_programs.size() < 32 &&
        std::ranges::find(pass_programs, pixel_program) == pass_programs.end()) {
        pass_programs.push_back(pixel_program);
    }
    for (const u64 hash : shader_hashes) {
        if (hash != 0 && pass_hashes.size() < 32 &&
            std::ranges::find(pass_hashes, hash) == pass_hashes.end()) {
            pass_hashes.push_back(hash);
        }
    }
}

void FrameDump::NoteConstants(std::string json) {
    pass_constants = std::move(json);
    pass_constants_set = true;
}

bool FrameDump::EndPass() {
    if (!pass_framebuffer) {
        return false;
    }
    pass_framebuffer = nullptr;
    std::string shaders;
    for (const u64 hash : pass_hashes) {
        shaders += fmt::format("{}\"{:016x}\"", shaders.empty() ? "" : ",", hash);
    }
    std::string programs;
    for (const u64 program : pass_programs) {
        programs += fmt::format("{}\"{:x}\"", programs.empty() ? "" : ",", program);
    }
    bool recorded = false;
    for (const Target& target : pass_targets) {
        recorded |= Record(
            target,
            fmt::format("{:04}_pass{:03}_d{}-{}_rt{}", sequence++, pass_index, pass_first_draw,
                        draws, target.slot),
            fmt::format(R"("kind":"pass","pass":{},"first_draw":{},"draw_count":{},"slot":{},"shaders":[{}],"ps_programs":[{}],"ps_cbufs":{{{}}})",
                        pass_index, pass_first_draw, draws - pass_first_draw, target.slot,
                        shaders, programs, pass_constants));
    }
    ++pass_index;
    return recorded;
}

void FrameDump::CaptureDispatch(u64 shader_hash, std::span<ImageView* const> written,
                                std::string_view written_buffers) {
    const u32 dispatch = dispatches++;
    // Every dispatch gets a line with the storage buffers it writes, images or not.
    manifest << fmt::format(
                    R"({{"kind":"dispatch_info","dispatch":{},"after_draws":{},"shaders":["{:016x}"],"ssbo_w":[{}]}})",
                    dispatch, draws, shader_hash, written_buffers)
             << '\n';
    for (u32 slot = 0; slot < written.size(); ++slot) {
        const auto target = written[slot] ? MakeTarget(*written[slot], slot) : std::nullopt;
        if (!target) {
            continue;
        }
        Record(*target,
               fmt::format("{:04}_dispatch{:04}_{:016x}_img{}", sequence++, dispatch, shader_hash,
                           slot),
               fmt::format(R"("kind":"dispatch","dispatch":{},"after_draws":{},"slot":{},"shaders":["{:016x}"])",
                           dispatch, draws, slot, shader_hash));
    }
}

void FrameDump::CapturePresent(ImageView& view) {
    if (const auto target = MakeTarget(view, 0)) {
        Record(*target, fmt::format("{:04}_present", sequence++),
               fmt::format(R"("kind":"present","after_draws":{})", draws));
    }
}

bool FrameDump::Record(const Target& target, std::string name, std::string tags) {
    name += fmt::format("_{}_{}x{}{}_{:x}", target.format, target.width, target.height,
                        target.depth > 1 ? fmt::format("x{}", target.depth) : "",
                        target.gpu_addr);
    if (target.msaa) {
        manifest << fmt::format(R"({{"name":"{}",{},"skipped":"msaa"}})", name, tags) << '\n';
        return false;
    }
    const size_t size = size_t{target.width} * target.height * target.depth *
                        VideoCore::Surface::BytesPerBlock(target.format);
    if (written_bytes + pending_bytes + size > SIZE_CAP) {
        truncated = true;
        return false;
    }
    vk::Buffer buffer = memory_allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        },
        MemoryUsage::Download);
    const VkBufferImageCopy copy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = static_cast<u32>(target.level),
            .baseArrayLayer = static_cast<u32>(target.layer),
            .layerCount = 1,
        },
        .imageOffset{0, 0, 0},
        .imageExtent{target.width, target.height, target.depth},
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = target.image, destination = *buffer, copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        cmdbuf.CopyImageToBuffer(image, VK_IMAGE_LAYOUT_GENERAL, destination, copy);
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                             VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                               after);
    });
    pending_bytes += size;
    // The slices are packed one after another: write them as one tall image.
    Target stacked = target;
    stacked.height *= target.depth;
    if (target.depth > 1) {
        tags += fmt::format(R"(,"depth":{},"slice_height":{})", target.depth, target.height);
    }
    stacked.depth = 1;
    readbacks.push_back({std::move(buffer), stacked, std::move(name), std::move(tags)});
    return true;
}

void FrameDump::DrainIfFull() {
    if (pending_bytes >= STAGING_BUDGET) {
        Drain();
    }
}

void FrameDump::Drain() {
    if (readbacks.empty()) {
        return;
    }
    scheduler.Finish();
    for (Readback& readback : readbacks) {
        readback.buffer.Invalidate();
        Write(readback, readback.buffer.Mapped());
    }
    readbacks.clear();
    pending_bytes = 0;
}

void FrameDump::Write(const Readback& readback, std::span<const u8> mapped) {
    const Target& target = readback.target;
    const u32 bytes_per_pixel = VideoCore::Surface::BytesPerBlock(target.format);
    const size_t pixels = size_t{target.width} * target.height;
    const auto data = mapped.first(pixels * bytes_per_pixel);
    WriteFile(readback.name + ".raw.zst",
              Common::Compression::CompressDataZSTDDefault(data.data(), data.size()));
    ++images;

    std::array<float, 4> pixel;
    const int channels = DecodePixel(target.format, data.data(), pixel);
    std::string stats;
    std::string png{"null"};
    if (channels != 0) {
        constexpr double inf = std::numeric_limits<double>::infinity();
        std::array<double, 4> lo{inf, inf, inf, inf};
        std::array<double, 4> hi{-inf, -inf, -inf, -inf};
        std::array<double, 4> sum{};
        std::array<u64, 4> nans{};
        std::array<u64, 4> infs{};
        const bool encode = IsFloat(target.format);
        std::vector<u8> rgb(pixels * 3);
        for (size_t index = 0; index < pixels; ++index) {
            DecodePixel(target.format, data.data() + index * bytes_per_pixel, pixel);
            bool finite = true;
            for (int c = 0; c < channels; ++c) {
                const double value = pixel[c];
                if (std::isnan(value)) {
                    ++nans[c];
                    finite = false;
                } else if (std::isinf(value)) {
                    ++infs[c];
                    finite = false;
                } else {
                    lo[c] = std::min(lo[c], value);
                    hi[c] = std::max(hi[c], value);
                    sum[c] += value;
                }
            }
            u8* const out = &rgb[index * 3];
            if (!finite) {
                out[0] = 255;
                out[1] = 0;
                out[2] = 255;
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                out[c] = PreviewByte(pixel[channels == 1 ? 0 : c], encode);
            }
        }
        std::array<double, 4> mean{};
        for (int c = 0; c < channels; ++c) {
            const u64 finite_count = pixels - nans[c] - infs[c];
            mean[c] = finite_count != 0 ? sum[c] / static_cast<double>(finite_count) : inf;
        }
        const auto first = [channels](const auto& values) {
            return std::span(values.data(), static_cast<size_t>(channels));
        };
        stats = fmt::format(R"(,"min":{},"max":{},"mean":{},"nan":{},"inf":{})",
                            JsonArray(first(lo)), JsonArray(first(hi)), JsonArray(first(mean)),
                            JsonArray(first(nans)), JsonArray(first(infs)));
        std::vector<u8> encoded;
        const int ok = stbi_write_png_to_func(
            [](void* context, void* bytes, int size) {
                auto* const out = static_cast<std::vector<u8>*>(context);
                const auto* const begin = static_cast<const u8*>(bytes);
                out->insert(out->end(), begin, begin + size);
            },
            &encoded, static_cast<int>(target.width), static_cast<int>(target.height), 3,
            rgb.data(), static_cast<int>(target.width * 3));
        if (ok != 0) {
            WriteFile(readback.name + ".png", encoded);
            png = fmt::format("\"{}.png\"", readback.name);
        }
    }
    manifest << fmt::format(
                    R"({{"name":"{}",{},"format":"{}","width":{},"height":{},"level":{},"layer":{},"gpu_addr":"{:#x}","rescaled":{},"raw":"{}.raw.zst","png":{}{}}})",
                    readback.name, readback.tags, target.format, target.width, target.height,
                    target.level, target.layer, target.gpu_addr, target.rescaled, readback.name,
                    png, stats)
             << '\n';
}

void FrameDump::WriteFile(const std::string& name, std::span<const u8> data) {
    std::ofstream file{capture_dir / name, std::ios::binary};
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    written_bytes += data.size();
}

void FrameDump::TickFrame() {
    if (!enabled || done) {
        return;
    }
    ++frame;
    if (capturing) {
        EndPass();
        Drain();
        manifest << fmt::format(
                        R"({{"kind":"summary","images":{},"passes":{},"draws":{},"dispatches":{},"bytes":{},"truncated":{}}})",
                        images, pass_index, draws, dispatches, written_bytes, truncated)
                 << '\n';
        manifest.close();
        capturing = false;
        done = true;
        LOG_INFO(Render_Vulkan,
                 "Frame dump done: {} images, {} passes, {} draws, {} dispatches, {} MiB in {}{}",
                 images, pass_index, draws, dispatches, written_bytes >> 20, capture_dir.string(),
                 truncated ? " (stopped at the 500 MiB cap)" : "");
        if (!trigger_file.empty()) {
            // File mode stays armed: each capture.now takes another frame into its own folder.
            done = false;
            truncated = false;
            readbacks.clear();
            pending_bytes = written_bytes = 0;
            sequence = images = 0;
            pass_framebuffer = nullptr;
            pass_targets.clear();
            pass_hashes.clear();
            pass_programs.clear();
            pass_constants.clear();
            pass_constants_set = false;
            pass_index = pass_first_draw = draws = dispatches = 0;
        }
        return;
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::error_code error;
    if (frame < trigger_frame && elapsed < trigger_seconds &&
        (trigger_file.empty() || frame % 30 != 0 ||
         !std::filesystem::exists(trigger_file, error))) {
        return;
    }
    capture_dir = directory;
    if (!trigger_file.empty()) {
        // Consume the trigger, and never overwrite an earlier capture.
        std::filesystem::remove(trigger_file, error);
        for (u32 index = 1;; ++index) {
            capture_dir = directory / fmt::format("capture-{:03}", index);
            if (!std::filesystem::exists(capture_dir, error)) {
                break;
            }
        }
    }
    std::filesystem::create_directories(capture_dir, error);
    manifest.open(capture_dir / "manifest.jsonl", std::ios::trunc);
    if (!manifest) {
        LOG_ERROR(Render_Vulkan, "Frame dump: cannot write {}", capture_dir.string());
        done = true;
        return;
    }
    manifest << fmt::format(R"({{"kind":"frame","frame":{},"seconds":{:.3f},"high_parts":{}}})", frame, elapsed, Settings::GPUHighParts())
             << '\n';
    capturing = true;
    LOG_INFO(Render_Vulkan, "Frame dump: capturing frame {} ({:.1f} s after boot) into {}", frame,
             elapsed, capture_dir.string());
}

} // namespace Vulkan
