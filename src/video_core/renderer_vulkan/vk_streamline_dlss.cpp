// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) in place of a game's DLSS evaluation.

#include <array>
#include <cstdlib>
#include <string_view>

#include "common/logging.h"
#include "video_core/dlss_evaluation.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkImageSubresourceRange COLOR_RANGE{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};
// Streamline clears the output before DLSS writes it.
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;

int EnvInt(const char* name, int fallback) {
    const char* const value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

// DLSS validates the render size against the mode's range: pick the mode for this ratio.
sl::DLSSMode ModeFor(VkExtent2D render, VkExtent2D upscale) {
    const f32 ratio = static_cast<f32>(render.width) / static_cast<f32>(upscale.width);
    if (ratio >= 0.99f) {
        return sl::DLSSMode::eDLAA;
    }
    if (ratio >= 0.65f) {
        return sl::DLSSMode::eMaxQuality;
    }
    if (ratio >= 0.57f) {
        return sl::DLSSMode::eBalanced;
    }
    if (ratio >= 0.49f) {
        return sl::DLSSMode::eMaxPerformance;
    }
    return sl::DLSSMode::eUltraPerformance;
}

// Preset K for every mode: NVIDIA's defaults for Performance/Ultra Performance (M, L) draw thin
// horizontal lines in Fire Emblem at 2x resolution scale. EDEN_NX2_DLSS_PRESET=J|K|L|M picks
// another model, EDEN_NX2_DLSS_PRESET=default leaves the choice to NVIDIA.
sl::DLSSPreset Preset() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_PRESET");
    if (value && value[0] >= 'J' && value[0] <= 'M') {
        return static_cast<sl::DLSSPreset>(static_cast<u32>(sl::DLSSPreset::ePresetJ) +
                                           static_cast<u32>(value[0] - 'J'));
    }
    if (value && std::string_view{value} == "default") {
        return sl::DLSSPreset::eDefault;
    }
    return sl::DLSSPreset::ePresetK;
}

sl::float4x4 Identity() {
    sl::float4x4 matrix;
    matrix.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    matrix.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    matrix.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    matrix.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    return matrix;
}

// Everything but the jitter, motion scale and reset, which vary. DLSS super resolution only reads
// the jitter, motion and depth conventions. Frame generation has no camera to read: motion vectors
// include it.
sl::Constants BaseConstants(f32 width, f32 height) {
    sl::Constants constants{};
    constants.cameraViewToClip = Identity();
    constants.clipToCameraView = Identity();
    constants.clipToLensClip = Identity();
    constants.clipToPrevClip = Identity();
    constants.prevClipToClip = Identity();
    constants.cameraPinholeOffset = {0.0f, 0.0f};
    constants.cameraPos = {0.0f, 0.0f, 0.0f};
    constants.cameraUp = {0.0f, 1.0f, 0.0f};
    constants.cameraRight = {1.0f, 0.0f, 0.0f};
    constants.cameraFwd = {0.0f, 0.0f, 1.0f};
    constants.cameraNear = 0.1f;
    constants.cameraFar = 1000.0f;
    constants.cameraFOV = 1.0f;
    constants.cameraAspectRatio = width / height;
    constants.depthInverted = sl::eTrue;
    constants.cameraMotionIncluded = sl::eTrue;
    constants.motionVectors3D = sl::eFalse;
    constants.motionVectorsJittered = sl::eFalse;
    return constants;
}

VkImageMemoryBarrier Barrier(const StreamlineDlssUpscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

sl::Resource Resource(VkImage image, VkImageView view, VkFormat format, VkImageLayout layout,
                      VkExtent2D size, VkImageUsageFlags usage) {
    sl::Resource resource{sl::ResourceType::eTex2d, image, nullptr, view,
                          static_cast<uint32_t>(layout)};
    resource.width = size.width;
    resource.height = size.height;
    resource.nativeFormat = static_cast<uint32_t>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0;
    resource.usage = usage;
    return resource;
}

} // Anonymous namespace

StreamlineDlssUpscaler::StreamlineDlssUpscaler(const Device& device_, Scheduler& scheduler_,
                                               MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!Streamline::IsDlssAvailable()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    const sl::Result result = Streamline::GetApi().get_feature_function(
        sl::kFeatureDLSS, "slDLSSSetOptions", set_options);
    if (result != sl::Result::eOk || !set_options) {
        LOG_ERROR(Render_Vulkan, "Streamline's DLSS has no slDLSSSetOptions (result {})",
                  static_cast<int>(result));
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

StreamlineDlssUpscaler::~StreamlineDlssUpscaler() {
    // Not scheduler.Finish(): this runs when the renderer goes away, after the game's channels are
    // released, and a submit would touch their state. Nothing more gets recorded by then.
    void(device.GetLogical().WaitIdle());
    if (Streamline::IsDlssAvailable()) {
        Streamline::GetApi().free_resources(sl::kFeatureDLSS, sl::ViewportHandle{0u});
    }
}

void StreamlineDlssUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
}

bool StreamlineDlssUpscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    const bool extent_changed = !output_image || in.render_size.width != render_extent.width ||
                                in.render_size.height != render_extent.height ||
                                in.upscale_size.width != upscale_extent.width ||
                                in.upscale_size.height != upscale_extent.height;
    const bool options_changed = extent_changed || in.hdr != hdr_input;
    const bool reset = in.reset || options_changed;
    if (extent_changed) {
        CreateOutput(in.upscale_size);
        render_extent = in.render_size;
        upscale_extent = in.upscale_size;
    }
    hdr_input = in.hdr;

    // Keep the existing HDR controls for linear input such as Fortune's Weave. Tonemapped
    // input such as Ocarina's already has its exposure applied.
    sl::DLSSOptions options{};
    options.mode = ModeFor(in.render_size, in.upscale_size);
    options.outputWidth = in.upscale_size.width;
    options.outputHeight = in.upscale_size.height;
    options.colorBuffersHDR =
        in.hdr && EnvInt("EDEN_NX2_DLSS_HDR", 1) != 0 ? sl::eTrue : sl::eFalse;
    options.useAutoExposure =
        in.hdr && EnvInt("EDEN_NX2_DLSS_AUTO_EXPOSURE", 1) != 0 ? sl::eTrue : sl::eFalse;
    const sl::DLSSPreset preset = Preset();
    options.dlaaPreset = preset;
    options.qualityPreset = preset;
    options.balancedPreset = preset;
    options.performancePreset = preset;
    options.ultraPerformancePreset = preset;
    options.ultraQualityPreset = preset;
    if (options_changed) {
        LOG_INFO(Render_Vulkan, "NVIDIA DLSS (Streamline): {}x{} -> {}x{}, mode {}, preset {}, {}",
                 in.render_size.width, in.render_size.height, in.upscale_size.width,
                 in.upscale_size.height, static_cast<int>(options.mode), static_cast<int>(preset),
                 options.colorBuffersHDR == sl::eTrue ? "HDR" : "LDR");
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, reset, options, options_changed, index = frame_index++,
                      output_image_handle = *output_image,
                      output_view_handle = *output_view](vk::CommandBuffer cmdbuf) {
        const Streamline::Api& api = Streamline::GetApi();
        const sl::ViewportHandle viewport{0u};
        // The caller draws the output this frame even when DLSS writes nothing to it: clear it
        // then, rather than show undefined contents (FSR 3.1 takes over from the next frame).
        const auto clear_output = [&] {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = output_image_handle,
                    .subresourceRange = range,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                   barrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT));
            cmdbuf.ClearColorImage(output_image_handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VkClearColorValue{}, range);
            cmdbuf.PipelineBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
        };
        sl::FrameToken* frame{};

        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // DLSS reads them as sampled images.
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };

        const f32 rw = static_cast<f32>(in.render_size.width);
        const f32 rh = static_cast<f32>(in.render_size.height);
        sl::Constants constants = BaseConstants(rw, rh);
        constants.jitterOffset = {in.jitter[0], in.jitter[1]};
        // Streamline wants motion in [-1,1] screen units; ours turns into render pixels.
        constants.mvecScale = {in.motion_vector_scale[0] / rw, in.motion_vector_scale[1] / rh};
        constants.cameraPinholeOffset = {0.0f, 0.0f};
        constants.cameraPos = {0.0f, 0.0f, 0.0f};
        constants.cameraUp = {0.0f, 1.0f, 0.0f};
        constants.cameraRight = {1.0f, 0.0f, 0.0f};
        constants.cameraFwd = {0.0f, 0.0f, 1.0f};
        constants.cameraNear = 0.1f;
        constants.cameraFar = 1000.0f;
        constants.cameraFOV = 1.0f;
        constants.cameraAspectRatio = rw / rh;
        constants.depthInverted = in.depth_inverted ? sl::eTrue : sl::eFalse;
        constants.cameraMotionIncluded = sl::eTrue;
        constants.motionVectors3D = sl::eFalse;
        constants.reset = reset ? sl::eTrue : sl::eFalse;

        sl::Resource color = Resource(in.color.image, in.color.view, in.color.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource depth = Resource(in.depth.image, in.depth.view, in.depth.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource motion = Resource(in.motion_vectors.image, in.motion_vectors.view,
                                       in.motion_vectors.format,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                       VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource output = Resource(output_image_handle, output_view_handle, OUTPUT_FORMAT,
                                       VK_IMAGE_LAYOUT_GENERAL, in.upscale_size, OUTPUT_USAGE);
        const sl::Extent render_area{0, 0, in.render_size.width, in.render_size.height};
        const sl::Extent output_area{0, 0, in.upscale_size.width, in.upscale_size.height};
        // Local tags: everything stays put until the evaluation is recorded.
        sl::ResourceTag color_tag{&color, sl::kBufferTypeScalingInputColor,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag depth_tag{&depth, sl::kBufferTypeDepth,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag motion_tag{&motion, sl::kBufferTypeMotionVectors,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag output_tag{&output, sl::kBufferTypeScalingOutputColor,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &output_area};
        const sl::BaseStructure* inputs[] = {&viewport, &color_tag, &depth_tag, &motion_tag,
                                             &output_tag};
        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        sl::Result api_result = sl::Result::eOk;
        const auto result = VideoCommon::RecordDlssEvaluation(
            [&] {
                if (!options_changed) {
                    return true;
                }
                const auto set = reinterpret_cast<PFun_slDLSSSetOptions*>(set_options);
                api_result = set(viewport, options);
                return api_result == sl::Result::eOk;
            },
            [&] {
                frame = Streamline::BeginEvaluation();
                if (!frame) {
                    api_result = api.get_new_frame_token(frame, &index);
                }
                return api_result == sl::Result::eOk && frame;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);
            },
            [&] {
                api_result = api.set_constants(constants, *frame, viewport);
                if (api_result == sl::Result::eOk && Streamline::IsFrameGenActive()) {
                    // The frame generator runs at present, when the game may have reused these:
                    // Streamline copies them here.
                    constexpr VkImageUsageFlags USAGE =
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                    sl::Resource tag_depth = depth;
                    sl::Resource tag_motion = motion;
                    tag_depth.usage = USAGE;
                    tag_motion.usage = USAGE;
                    const std::array tags{
                        sl::ResourceTag{&tag_depth, sl::kBufferTypeDepth,
                                        sl::ResourceLifecycle::eOnlyValidNow, &render_area},
                        sl::ResourceTag{&tag_motion, sl::kBufferTypeMotionVectors,
                                        sl::ResourceLifecycle::eOnlyValidNow, &render_area},
                    };
                    const sl::Result tagged = api.set_tag_for_frame(
                        *frame, viewport, tags.data(), static_cast<uint32_t>(tags.size()),
                        reinterpret_cast<sl::CommandBuffer*>(*cmdbuf));
                    if (tagged != sl::Result::eOk) {
                        Streamline::FrameGenFailed("slSetTagForFrame", static_cast<int>(tagged));
                    }
                }
                return api_result == sl::Result::eOk;
            },
            [&] {
                api_result = api.evaluate_feature(sl::kFeatureDLSS, *frame, inputs, std::size(inputs),
                                                  reinterpret_cast<sl::CommandBuffer*>(*cmdbuf));
                return api_result == sl::Result::eOk;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                                       to_general);
            });
        if (result != VideoCommon::DlssEvaluationResult::Succeeded && !failed.exchange(true)) {
            LOG_ERROR(Render_Vulkan,
                      "NVIDIA DLSS recording failed at stage {} (Streamline result {}); using FSR 3.1",
                      static_cast<int>(result), static_cast<int>(api_result));
        }
        if (result != VideoCommon::DlssEvaluationResult::Succeeded) {
            clear_output();
        }
    });
    // DLSS binds its own pipelines and descriptors on the command buffer.
    scheduler.InvalidateState();
    return true;
}

StreamlineFrameTagger::StreamlineFrameTagger(const Device& device_, Scheduler& scheduler_,
                                             MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {}

StreamlineFrameTagger::~StreamlineFrameTagger() {
    void(device.GetLogical().WaitIdle());
}

void StreamlineFrameTagger::CreateMotion(VkExtent2D size) {
    if (motion_image) {
        scheduler.Finish(); // the old image may still be in use
    }
    constexpr VkFormat FORMAT = VK_FORMAT_R16G16_SFLOAT;
    motion_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = FORMAT,
        .extent = {size.width, size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    motion_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *motion_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = FORMAT,
        .components{},
        .subresourceRange = COLOR_RANGE,
    });
    extent = size;
    LOG_INFO(Render_Vulkan, "DLSS frame generation inputs: {}x{} depth, blank motion vectors",
             size.width, size.height);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *motion_image](vk::CommandBuffer cmdbuf) {
        const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access,
                                 VkImageLayout old_layout, VkImageLayout new_layout) {
            return VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = src_access,
                .dstAccessMask = dst_access,
                .oldLayout = old_layout,
                .newLayout = new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange = COLOR_RANGE,
            };
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               barrier(0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        cmdbuf.ClearColorImage(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VkClearColorValue{},
                               COLOR_RANGE);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                               barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    });
}

void StreamlineFrameTagger::Tag(const StreamlineDlssUpscaler::Image& depth, VkExtent2D size) {
    const bool resized = size.width != extent.width || size.height != extent.height;
    if (resized) {
        CreateMotion(size);
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([depth, size, resized, motion_image_handle = *motion_image,
                      motion_view_handle = *motion_view](vk::CommandBuffer cmdbuf) {
        if (Streamline::FrameHasInputs()) {
            return; // a DLSS evaluation tagged this frame
        }
        sl::FrameToken* const frame = Streamline::BeginEvaluation();
        if (!frame) {
            return;
        }
        const Streamline::Api& api = Streamline::GetApi();
        const sl::ViewportHandle viewport{0u};
        sl::Constants constants =
            BaseConstants(static_cast<f32>(size.width), static_cast<f32>(size.height));
        constants.jitterOffset = {0.0f, 0.0f};
        constants.mvecScale = {1.0f, 1.0f};
        constants.reset = resized ? sl::eTrue : sl::eFalse;
        if (const sl::Result result = api.set_constants(constants, *frame, viewport);
            result != sl::Result::eOk) {
            Streamline::FrameGenFailed("slSetConstants", static_cast<int>(result));
            return;
        }
        constexpr VkAccessFlags DEPTH_WRITE = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        constexpr VkAccessFlags DEPTH_READ =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                               Barrier(depth, DEPTH_WRITE, DEPTH_READ, VK_IMAGE_LAYOUT_GENERAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        constexpr VkImageUsageFlags USAGE =
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        sl::Resource depth_resource =
            Resource(depth.image, depth.view, depth.format,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, size, USAGE);
        sl::Resource motion_resource =
            Resource(motion_image_handle, motion_view_handle, VK_FORMAT_R16G16_SFLOAT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, size, USAGE);
        const sl::Extent area{0, 0, size.width, size.height};
        const std::array tags{
            sl::ResourceTag{&depth_resource, sl::kBufferTypeDepth,
                            sl::ResourceLifecycle::eOnlyValidNow, &area},
            sl::ResourceTag{&motion_resource, sl::kBufferTypeMotionVectors,
                            sl::ResourceLifecycle::eOnlyValidNow, &area},
        };
        const sl::Result tagged = api.set_tag_for_frame(
            *frame, viewport, tags.data(), static_cast<uint32_t>(tags.size()),
            reinterpret_cast<sl::CommandBuffer*>(*cmdbuf));
        if (tagged != sl::Result::eOk) {
            Streamline::FrameGenFailed("slSetTagForFrame", static_cast<int>(tagged));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                               Barrier(depth, DEPTH_READ, DEPTH_READ | DEPTH_WRITE,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_IMAGE_LAYOUT_GENERAL));
    });
    scheduler.InvalidateState();
}

} // namespace Vulkan
