// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA Streamline. Only Streamline's MIT-licensed
// headers are built in: NVIDIA's DLLs are loaded at run time from the user's PC, and none ship
// with the emulator.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Streamline {

/// NVIDIA's files DLSS needs, as found on this PC. Paths are empty when missing.
struct DlssFiles {
    std::filesystem::path folder;     ///< Where the loader is, or where it should go
    std::filesystem::path interposer; ///< sl.interposer.dll (Streamline's loader)
    std::filesystem::path model;      ///< nvngx_dlss.dll next to the loader
    std::filesystem::path app_model;  ///< Newest DLSS the NVIDIA app downloaded
    std::string interposer_version;
    std::string model_version;
    std::string app_model_version;
};

[[nodiscard]] DlssFiles FindDlssFiles();

/// The user folder's "streamline" folder.
[[nodiscard]] std::filesystem::path UserFolder();

/// Streamline's own modules carry a second, Streamline-only signature.
[[nodiscard]] bool IsStreamlineModule(const std::filesystem::path& path);

/// A valid Authenticode signature from NVIDIA Corporation (DLSS models).
[[nodiscard]] bool IsSignedByNvidia(const std::filesystem::path& path);

/// When the NVIDIA DLSS setting is on, loads the user's sl.interposer.dll (checking NVIDIA's
/// signature) and starts Streamline for DLSS. Call before creating the renderer's instance.
bool Start();

/// Whether Streamline is running and waiting for the renderer's device.
[[nodiscard]] bool WantsDevice();

/// Instance and device extensions DLSS asks for (empty when Streamline isn't running).
[[nodiscard]] std::vector<std::string> InstanceExtensions();
[[nodiscard]] std::vector<std::string> DeviceExtensions();

/// Queues DLSS Frame Generation needs on top of the host's.
struct FrameGenQueues {
    u32 graphics{};
    u32 compute{};
    u32 optical_flow{};
};

/// What the device set up for DLSS Frame Generation (EDEN_NX2_DLSS_G=1).
struct FrameGenDevice {
    bool enabled{};
    u32 compute_family{};
    u32 optical_flow_family{};
    bool native_optical_flow{};
};

/// A finished frame on its way to the screen.
struct PresentedFrame {
    void* token{};  ///< sl::FrameToken of the frame, null without frame generation
    bool generate{}; ///< DLSS ran for this frame, so one can be generated after it
};

/// Device extensions frame generation and Reflex ask for (empty when it wasn't asked for).
[[nodiscard]] std::vector<std::string> FrameGenDeviceExtensions();

[[nodiscard]] FrameGenQueues FrameGenQueuesWanted();

/// Hands the renderer's device to Streamline and routes the swapchain calls through it, as
/// Streamline's manual hooking needs. Stops Streamline and returns false when DLSS can't run;
/// without frame generation DLSS still runs.
bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  u32 graphics_family, const FrameGenDevice& frame_gen, vk::DeviceDispatch& dld);

/// Whether frame generation runs on the attached device.
[[nodiscard]] bool IsFrameGenActive();

/// Streamline must see the window's surface created, which it can only do with a device: replaces
/// `surface` with one made through it. Call after AttachDevice, before creating the swapchain.
bool RecreateSurface(const vk::Instance& instance, vk::SurfaceKHR& surface, void* window);

/// Ends the frame on the render worker, opening the next one.
[[nodiscard]] PresentedFrame FinishFrame();

/// Around the present of `frame`, on the present thread.
void PresentStart(const PresentedFrame& frame);
void PresentEnd(const PresentedFrame& frame);

/// Turns generation off until the next present; call before recreating the swapchain.
void PauseFrameGen();

/// Stops generating frames for this session, keeping DLSS.
void FrameGenFailed(const char* what, int result);

/// Whether DLSS can run on the attached device.
[[nodiscard]] bool IsDlssAvailable();

/// Shuts Streamline down; call before the device is destroyed.
void Stop();

} // namespace Vulkan::Streamline
