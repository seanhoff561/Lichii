// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA Streamline, with Streamline's manual hooking.

#include <algorithm>
#include <utility>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread_worker.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"

#include <sl_security.h>

#include <softpub.h>
#include <wintrust.h>

namespace Vulkan::Streamline {
namespace {

// A custom engine names itself with its own project id; an application id from NVIDIA is only
// needed without an engine type. Keep it fixed: the driver may key behaviour off it.
constexpr const char* PROJECT_ID = "8d3e2f9a-6b41-4c7e-9a52-1f0c3b7d5e64";
constexpr const wchar_t* INTERPOSER = L"sl.interposer.dll";
constexpr const wchar_t* MODEL = L"nvngx_dlss.dll";
constexpr const wchar_t* FRAME_GEN_MODEL = L"nvngx_dlssg.dll";
// The first DLSS with preset K.
constexpr u32 MIN_DLSS_MAJOR = 310;

// Frames in a row with DLSS before generation turns on, so a menu that flickers between frames
// with and without DLSS doesn't toggle it.
constexpr u32 FRAME_GEN_ON_AFTER = 3;
// Real frames between reports in the log.
constexpr u32 FRAME_GEN_REPORT_EVERY = 600;

using Version = std::array<u32, 4>;

struct FrameGenState {
    bool wanted{};
    bool active{};
    std::atomic<bool> failed{};
    std::vector<std::string> device_extensions;
    FrameGenQueues queues{};
    PFun_slDLSSGSetOptions* set_options{};
    PFun_slDLSSGGetState* get_state{};
    PFun_slPCLSetMarker* set_marker{};
    PFun_slReflexSetOptions* reflex_options{};
    PFun_slReflexSleep* reflex_sleep{};

    // Render worker only.
    sl::FrameToken* open_frame{};
    u32 next_index{1};
    bool evaluated{};

    // Present thread only.
    bool mode_on{};
    bool reflex_on{};
    u32 dlss_streak{};
    u32 frames{};
    u32 toggles{};
    u32 presented{};
    u32 last_status{};

    /// Reflex wants a sleep every frame, which must not hold up rendering or presenting.
    std::unique_ptr<Common::ThreadWorker> sleep_worker;
    std::atomic<bool> sleeping{};
};

struct State {
    HMODULE module{};
    Api api{};
    std::wstring plugin_dir;
    std::wstring model_dir;
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    bool started{};
    bool attached{};
    FrameGenState frame_gen;
};

State& GetState() {
    static State state;
    return state;
}

int LogLevel() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_LOG");
    return value ? std::atoi(value) : 0;
}

bool FrameGenSwitch() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_G");
    return value && std::atoi(value) != 0;
}

void LogFromStreamline(sl::LogType type, const char* message) {
    std::string_view text{message ? message : ""};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    switch (type) {
    case sl::LogType::eError:
        LOG_ERROR(Render_Vulkan, "Streamline: {}", text);
        break;
    case sl::LogType::eWarn:
        LOG_WARNING(Render_Vulkan, "Streamline: {}", text);
        break;
    default:
        if (LogLevel() > 0) {
            LOG_INFO(Render_Vulkan, "Streamline: {}", text);
        }
        break;
    }
}

/// Where the user may put sl.interposer.dll: EDEN_STREAMLINE_DIR, the user folder's "streamline"
/// folder, then a "streamline" folder next to the executable.
std::vector<std::filesystem::path> SearchDirs() {
    std::vector<std::filesystem::path> dirs;
    if (const char* const dir = std::getenv("EDEN_STREAMLINE_DIR"); dir && *dir) {
        dirs.emplace_back(dir);
    }
    dirs.push_back(UserFolder());
    dirs.push_back(Common::FS::GetExeDirectory() / "streamline");
    return dirs;
}

template <typename T>
bool Resolve(HMODULE module, const char* name, T*& function) {
    function = reinterpret_cast<T*>(GetProcAddress(module, name));
    if (!function) {
        LOG_ERROR(Render_Vulkan, "sl.interposer.dll has no {}", name);
    }
    return function != nullptr;
}

std::optional<Version> ReadVersion(const std::filesystem::path& path) {
    DWORD handle = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &handle);
    if (size == 0) {
        return std::nullopt;
    }
    std::vector<u8> data(size);
    VS_FIXEDFILEINFO* info{};
    UINT length = 0;
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data()) ||
        !VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &length) || !info) {
        return std::nullopt;
    }
    return Version{HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS),
                   HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS)};
}

std::string VersionText(const std::filesystem::path& path) {
    const std::optional<Version> version = path.empty() ? std::nullopt : ReadVersion(path);
    if (!version) {
        return {};
    }
    const Version& v = *version;
    return v[3] == 0 ? fmt::format("{}.{}.{}", v[0], v[1], v[2])
                     : fmt::format("{}.{}.{}.{}", v[0], v[1], v[2], v[3]);
}

/// The NVIDIA app keeps every model it downloads under ProgramData; pick the newest.
std::filesystem::path FindAppModel(const wchar_t* feature = L"dlss") {
    const wchar_t* const program_data = _wgetenv(L"ProgramData");
    if (!program_data) {
        return {};
    }
    const std::filesystem::path root =
        std::filesystem::path{program_data} / L"NVIDIA" / L"NGX" / L"models" / feature / L"versions";
    std::filesystem::path best_path;
    Version best{};
    std::error_code ec;
    for (const auto& version_dir : std::filesystem::directory_iterator(root, ec)) {
        std::error_code files_ec;
        for (const auto& file :
             std::filesystem::directory_iterator(version_dir.path() / L"files", files_ec)) {
            if (file.path().extension() != L".bin") {
                continue;
            }
            const std::optional<Version> version = ReadVersion(file.path());
            if (version && (*version)[0] >= MIN_DLSS_MAJOR && *version > best) {
                best = *version;
                best_path = file.path();
            }
        }
    }
    return best_path;
}

/// Copies the NVIDIA app's DLSS into our folder as nvngx_dlss.dll, the name Streamline loads.
std::filesystem::path UseAppModel(const DlssFiles& files) {
    const std::filesystem::path dir = UserFolder() / L"nvidia-app";
    const std::filesystem::path target = dir / MODEL;
    std::error_code ec;
    const bool current = std::filesystem::is_regular_file(target, ec) &&
                         std::filesystem::file_size(target, ec) ==
                             std::filesystem::file_size(files.app_model, ec) &&
                         ReadVersion(target) == ReadVersion(files.app_model);
    if (!current) {
        if (!IsSignedByNvidia(files.app_model)) {
            LOG_WARNING(Render_Vulkan, "The NVIDIA app's DLSS {} is not signed by NVIDIA",
                        Common::FS::PathToUTF8String(files.app_model));
            return {};
        }
        std::filesystem::create_directories(dir, ec);
        std::filesystem::copy_file(files.app_model, target,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            LOG_WARNING(Render_Vulkan, "Could not copy the NVIDIA app's DLSS to {}: {}",
                        Common::FS::PathToUTF8String(target), ec.message());
            return {};
        }
    }
    if (!IsSignedByNvidia(target)) {
        LOG_WARNING(Render_Vulkan, "Cached DLSS {} is not signed by NVIDIA",
                    Common::FS::PathToUTF8String(target));
        return {};
    }
    return dir;
}

/// NGX only looks for frame generation's model next to DLSS: copies the NVIDIA app's newest there.
void UseAppFrameGenModel(const std::filesystem::path& dir) {
    const std::filesystem::path source = FindAppModel(L"dlssg");
    if (source.empty()) {
        LOG_WARNING(Render_Vulkan, "The NVIDIA app has no DLSS frame generation model");
        return;
    }
    const std::filesystem::path target = dir / FRAME_GEN_MODEL;
    std::error_code ec;
    if (std::filesystem::is_regular_file(target, ec) && ReadVersion(target) == ReadVersion(source) &&
        IsSignedByNvidia(target)) {
        return;
    }
    if (!IsSignedByNvidia(source)) {
        LOG_WARNING(Render_Vulkan, "The NVIDIA app's frame generation model {} is not signed by NVIDIA",
                    Common::FS::PathToUTF8String(source));
        return;
    }
    std::filesystem::create_directories(dir, ec);
    std::filesystem::copy_file(source, target, std::filesystem::copy_options::overwrite_existing,
                               ec);
    if (ec) {
        LOG_WARNING(Render_Vulkan, "Could not copy the frame generation model to {}: {}",
                    Common::FS::PathToUTF8String(target), ec.message());
        return;
    }
    LOG_INFO(Render_Vulkan, "Using the NVIDIA app's frame generation model {}", VersionText(target));
}

bool Load(State& state, const std::filesystem::path& path) {
    if (!IsStreamlineModule(path)) {
        LOG_ERROR(Render_Vulkan, "{} is not signed by NVIDIA; not loading it",
                  Common::FS::PathToUTF8String(path));
        return false;
    }
    state.module = LoadLibraryW(path.c_str());
    if (!state.module) {
        LOG_ERROR(Render_Vulkan, "Could not load {} (error {})", Common::FS::PathToUTF8String(path),
                  GetLastError());
        return false;
    }
    Api& api = state.api;
    const HMODULE m = state.module;
    if (Resolve(m, "slInit", api.init) && Resolve(m, "slShutdown", api.shutdown) &&
        Resolve(m, "slIsFeatureSupported", api.is_feature_supported) &&
        Resolve(m, "slGetFeatureRequirements", api.get_feature_requirements) &&
        Resolve(m, "slGetFeatureVersion", api.get_feature_version) &&
        Resolve(m, "slGetFeatureFunction", api.get_feature_function) &&
        Resolve(m, "slGetNewFrameToken", api.get_new_frame_token) &&
        Resolve(m, "slSetConstants", api.set_constants) &&
        Resolve(m, "slSetTagForFrame", api.set_tag_for_frame) &&
        Resolve(m, "slEvaluateFeature", api.evaluate_feature) &&
        Resolve(m, "slFreeResources", api.free_resources) &&
        Resolve(m, "slSetVulkanInfo", api.set_vulkan_info)) {
        return true;
    }
    FreeLibrary(state.module);
    state.module = nullptr;
    state.api = {};
    return false;
}

std::vector<std::string> Names(uint32_t count, const char** names) {
    std::vector<std::string> result;
    for (uint32_t i = 0; i < count; ++i) {
        result.emplace_back(names[i]);
    }
    return result;
}

void AddMissing(std::vector<std::string>& to, const std::vector<std::string>& names) {
    for (const std::string& name : names) {
        if (std::ranges::find(to, name) == to.end()) {
            to.push_back(name);
        }
    }
}

/// Reads what frame generation and Reflex need; leaves frame generation off when they can't run.
void QueryFrameGen(State& state) {
    FrameGenState& fg = state.frame_gen;
    const Api& api = state.api;
    sl::FeatureRequirements frame_gen{};
    sl::FeatureRequirements reflex{};
    if (const sl::Result result = api.get_feature_requirements(sl::kFeatureDLSS_G, frame_gen);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "Streamline found no frame generation plugin (result {})",
                    static_cast<int>(result));
        return;
    }
    if (const sl::Result result = api.get_feature_requirements(sl::kFeatureReflex, reflex);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "Streamline found no Reflex plugin (result {})",
                    static_cast<int>(result));
        return;
    }
    if ((static_cast<uint32_t>(frame_gen.flags) &
         static_cast<uint32_t>(sl::FeatureRequirementFlags::eVulkanSupported)) == 0) {
        LOG_WARNING(Render_Vulkan, "This frame generation plugin does not support Vulkan");
        return;
    }
    fg.queues = {
        .graphics = frame_gen.vkNumGraphicsQueuesRequired,
        .compute = frame_gen.vkNumComputeQueuesRequired,
        .optical_flow = frame_gen.vkNumOpticalFlowQueuesRequired,
    };
    AddMissing(state.instance_extensions,
               Names(frame_gen.vkNumInstanceExtensions, frame_gen.vkInstanceExtensions));
    AddMissing(state.instance_extensions,
               Names(reflex.vkNumInstanceExtensions, reflex.vkInstanceExtensions));
    AddMissing(fg.device_extensions,
               Names(frame_gen.vkNumDeviceExtensions, frame_gen.vkDeviceExtensions));
    AddMissing(fg.device_extensions, Names(reflex.vkNumDeviceExtensions, reflex.vkDeviceExtensions));
    fg.wanted = true;
    LOG_INFO(Render_Vulkan, "DLSS frame generation wants {} graphics, {} compute and {} optical "
                            "flow queues",
             fg.queues.graphics, fg.queues.compute, fg.queues.optical_flow);
}

template <typename T>
bool FeatureFunction(sl::Feature feature, const char* name, T*& function) {
    void* address{};
    const sl::Result result = GetState().api.get_feature_function(feature, name, address);
    function = reinterpret_cast<T*>(address);
    if (result != sl::Result::eOk || !function) {
        LOG_WARNING(Render_Vulkan, "Streamline has no {} (result {})", name,
                    static_cast<int>(result));
        return false;
    }
    return true;
}

/// Looks up frame generation on the attached device. DLSS runs without it.
bool AttachFrameGen(State& state, VkPhysicalDevice physical) {
    FrameGenState& fg = state.frame_gen;
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physical;
    for (const sl::Feature feature :
         {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL}) {
        if (const sl::Result result = state.api.is_feature_supported(feature, adapter);
            result != sl::Result::eOk) {
            LOG_WARNING(Render_Vulkan,
                        "Streamline feature {} is not supported here (result {}); no frame "
                        "generation (HAGS must be on)",
                        static_cast<u32>(feature), static_cast<int>(result));
            return false;
        }
    }
    return FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGSetOptions", fg.set_options) &&
           FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGGetState", fg.get_state) &&
           FeatureFunction(sl::kFeaturePCL, "slPCLSetMarker", fg.set_marker) &&
           FeatureFunction(sl::kFeatureReflex, "slReflexSetOptions", fg.reflex_options) &&
           FeatureFunction(sl::kFeatureReflex, "slReflexSleep", fg.reflex_sleep);
}

void Mark(sl::PCLMarker marker, const sl::FrameToken& frame) {
    GetState().frame_gen.set_marker(marker, frame);
}

/// Starts the next frame; game work up to its DLSS evaluation counts as its simulation.
void OpenFrame(State& state) {
    FrameGenState& fg = state.frame_gen;
    const uint32_t index = fg.next_index++;
    sl::FrameToken* token{};
    fg.open_frame = nullptr;
    fg.evaluated = false;
    if (const sl::Result result = state.api.get_new_frame_token(token, &index);
        result != sl::Result::eOk || !token) {
        FrameGenFailed("slGetNewFrameToken", static_cast<int>(result));
        return;
    }
    fg.open_frame = token;
    Mark(sl::PCLMarker::eSimulationStart, *token);
}

void SetFrameGenMode(State& state, bool on) {
    FrameGenState& fg = state.frame_gen;
    sl::DLSSGOptions options{};
    options.mode = on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    options.numFramesToGenerate = 1;
    options.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    if (std::getenv("EDEN_NX2_DLSS_G_ONLY")) {
        options.flags |= sl::DLSSGFlags::eShowOnlyInterpolatedFrame; // to judge generated frames
    }
    // The depth and motion tags are copied when tagged, so rendering need not wait for DLSS-G.
    static const bool parallel = !std::getenv("EDEN_NX2_DLSS_G_BLOCK");
    options.queueParallelismMode = parallel ? sl::DLSSGQueueParallelismMode::eBlockNoClientQueues
                                            : sl::DLSSGQueueParallelismMode::eBlockPresentingClientQueue;
    if (const sl::Result result = fg.set_options(sl::ViewportHandle{0u}, options);
        result != sl::Result::eOk) {
        FrameGenFailed("slDLSSGSetOptions", static_cast<int>(result));
        return;
    }
    fg.mode_on = on;
    ++fg.toggles;
    // Often at first, as a game going between its menus and 3D turns this on and off.
    static u32 logged{};
    if (logged < 20) {
        ++logged;
        LOG_INFO(Render_Vulkan, "DLSS frame generation {}", on ? "on" : "off");
    }
}

/// What the generator says about itself, as it fails silently otherwise.
void ReportFrameGen(State& state) {
    FrameGenState& fg = state.frame_gen;
    ++fg.frames;
    if (fg.frames % 60 != 0) {
        return;
    }
    sl::DLSSGState frame_gen{};
    if (const sl::Result result = fg.get_state(sl::ViewportHandle{0u}, frame_gen, nullptr);
        result != sl::Result::eOk) {
        static bool reported{};
        if (!std::exchange(reported, true)) {
            LOG_WARNING(Render_Vulkan, "slDLSSGGetState returned {}", static_cast<int>(result));
        }
        return;
    }
    fg.presented += frame_gen.numFramesActuallyPresented;
    const u32 status = static_cast<u32>(frame_gen.status);
    if (status != fg.last_status) {
        fg.last_status = status;
        LOG_WARNING(Render_Vulkan, "DLSS frame generation status: {:#x}", status);
    }
    if (fg.frames % FRAME_GEN_REPORT_EVERY == 0) {
        LOG_INFO(Render_Vulkan,
                 "DLSS frame generation: {} presents for {} frames, {} mode changes, status {:#x}",
                 fg.presented, FRAME_GEN_REPORT_EVERY, fg.toggles, status);
        fg.presented = 0;
        fg.toggles = 0;
    }
}

} // Anonymous namespace

std::filesystem::path UserFolder() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::StreamlineDir);
}

bool IsStreamlineModule(const std::filesystem::path& path) {
    return sl::security::verifyEmbeddedSignature(path.c_str());
}

bool IsSignedByNvidia(const std::filesystem::path& path) {
    const std::wstring file_path = std::filesystem::absolute(path).wstring();
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof(file);
    file.pcwszFilePath = file_path.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof(data);
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    bool valid = WinVerifyTrust(nullptr, &action, &data) == ERROR_SUCCESS;
    if (valid) {
        CRYPT_PROVIDER_DATA* const provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        CRYPT_PROVIDER_SGNR* const signer =
            provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
        CRYPT_PROVIDER_CERT* const cert = signer ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[128]{};
        valid = cert &&
                CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name,
                                   static_cast<DWORD>(std::size(name))) > 1 &&
                std::wstring_view{name} == L"NVIDIA Corporation";
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return valid;
}

DlssFiles FindDlssFiles() {
    DlssFiles files;
    files.folder = UserFolder();
    std::error_code ec;
    for (const std::filesystem::path& dir : SearchDirs()) {
        if (std::filesystem::is_regular_file(dir / INTERPOSER, ec)) {
            files.folder = dir;
            files.interposer = dir / INTERPOSER;
            break;
        }
    }
    if (std::filesystem::is_regular_file(files.folder / MODEL, ec)) {
        files.model = files.folder / MODEL;
    }
    files.app_model = FindAppModel();
    files.interposer_version = VersionText(files.interposer);
    files.model_version = VersionText(files.model);
    files.app_model_version = VersionText(files.app_model);
    return files;
}

bool Start() {
    State& state = GetState();
    if (state.started) {
        return true;
    }
    if (!Settings::values.nvidia_dlss.GetValue()) {
        return false;
    }
    if (!state.module) {
        const DlssFiles files = FindDlssFiles();
        if (!files.model.empty() && !IsSignedByNvidia(files.model)) {
            LOG_WARNING(Render_Vulkan, "DLSS model {} is not signed by NVIDIA; using FSR 3.1",
                        Common::FS::PathToUTF8String(files.model));
            return false;
        }
        if (files.interposer.empty()) {
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS is on, but sl.interposer.dll (NVIDIA Streamline) is not in "
                        "{} (the Graphics settings can download it); using FSR 3.1",
                        Common::FS::PathToUTF8String(files.folder));
            return false;
        }
        if (!Load(state, files.interposer)) {
            return false;
        }
        state.plugin_dir = files.folder.wstring();
        state.model_dir.clear();
        if (files.model.empty() && !files.app_model.empty()) {
            if (const std::filesystem::path dir = UseAppModel(files); !dir.empty()) {
                state.model_dir = dir.wstring();
                LOG_INFO(Render_Vulkan, "Using the NVIDIA app's DLSS {}", files.app_model_version);
            }
        } else if (files.model.empty()) {
            LOG_WARNING(Render_Vulkan, "No nvngx_dlss.dll in {} and no DLSS from the NVIDIA app",
                        Common::FS::PathToUTF8String(files.folder));
        }
        if (FrameGenSwitch()) {
            UseAppFrameGenModel(state.model_dir.empty() ? files.folder
                                                        : std::filesystem::path{state.model_dir});
        }
    }

    // Streamline's plugins come from the loader's folder or the NVIDIA app's downloads (OTA); the
    // DLSS model from the loader's folder, else the NVIDIA app's copy.
    const wchar_t* plugin_paths[] = {state.plugin_dir.c_str(), state.model_dir.c_str()};
    std::vector<sl::Feature> features{sl::kFeatureDLSS};
    if (FrameGenSwitch()) {
        features.insert(features.end(),
                        {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL});
    }
    sl::Preferences preferences{};
    preferences.logLevel = LogLevel() >= 2 ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    preferences.logMessageCallback = LogFromStreamline;
    preferences.pathsToPlugins = plugin_paths;
    preferences.numPathsToPlugins = state.model_dir.empty() ? 1 : 2;
    preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                        sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eAllowOTA |
                        sl::PreferenceFlags::eLoadDownloadedPlugins |
                        sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    preferences.featuresToLoad = features.data();
    preferences.numFeaturesToLoad = static_cast<uint32_t>(features.size());
    preferences.engine = sl::EngineType::eCustom;
    preferences.engineVersion = "1.0";
    preferences.projectId = PROJECT_ID;
    preferences.renderAPI = sl::RenderAPI::eVulkan;
    const Api& api = state.api;
    if (const sl::Result result = api.init(preferences, sl::kSDKVersion);
        result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "NVIDIA Streamline failed to start (result {}); using FSR 3.1",
                  static_cast<int>(result));
        return false;
    }
    sl::FeatureRequirements requirements{};
    if (const sl::Result result = api.get_feature_requirements(sl::kFeatureDLSS, requirements);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan,
                    "Streamline found no DLSS plugin (result {}); put sl.dlss.dll next to "
                    "sl.interposer.dll or install the NVIDIA app. Using FSR 3.1",
                    static_cast<int>(result));
        api.shutdown();
        return false;
    }
    if (requirements.vkNumComputeQueuesRequired != 0 ||
        requirements.vkNumGraphicsQueuesRequired != 0) {
        LOG_WARNING(Render_Vulkan, "This DLSS wants extra Vulkan queues; using FSR 3.1");
        api.shutdown();
        return false;
    }
    state.instance_extensions =
        Names(requirements.vkNumInstanceExtensions, requirements.vkInstanceExtensions);
    state.device_extensions =
        Names(requirements.vkNumDeviceExtensions, requirements.vkDeviceExtensions);
    // The KHR and EXT buffer device address extensions can't both be enabled.
    const auto& device_extensions = state.device_extensions;
    if (std::ranges::find(device_extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) !=
        device_extensions.end()) {
        std::erase(state.device_extensions, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }
    state.frame_gen.wanted = false;
    state.frame_gen.device_extensions.clear();
    if (FrameGenSwitch()) {
        QueryFrameGen(state);
    }
    sl::FeatureVersion version{};
    api.get_feature_version(sl::kFeatureDLSS, version);
    LOG_INFO(Render_Vulkan, "NVIDIA Streamline {} started from {} (DLSS {})",
             version.versionSL.toStr(),
             Common::FS::PathToUTF8String(std::filesystem::path{state.plugin_dir}),
             version.versionNGX.toStr());
    state.started = true;
    return true;
}

bool WantsDevice() {
    const State& state = GetState();
    return state.started && !state.attached;
}

std::vector<std::string> InstanceExtensions() {
    const State& state = GetState();
    return state.started ? state.instance_extensions : std::vector<std::string>{};
}

std::vector<std::string> DeviceExtensions() {
    const State& state = GetState();
    return state.started ? state.device_extensions : std::vector<std::string>{};
}

std::vector<std::string> FrameGenDeviceExtensions() {
    const State& state = GetState();
    return state.started && state.frame_gen.wanted ? state.frame_gen.device_extensions
                                                   : std::vector<std::string>{};
}

FrameGenQueues FrameGenQueuesWanted() {
    const State& state = GetState();
    return state.started && state.frame_gen.wanted ? state.frame_gen.queues : FrameGenQueues{};
}

bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  u32 graphics_family, const FrameGenDevice& frame_gen, vk::DeviceDispatch& dld) {
    State& state = GetState();
    if (!WantsDevice()) {
        return false;
    }
    sl::VulkanInfo info{};
    info.device = device;
    info.instance = instance;
    info.physicalDevice = physical;
    info.computeQueueFamily = graphics_family;
    info.graphicsQueueFamily = graphics_family;
    if (frame_gen.enabled) {
        // Streamline's own queues: after the host's graphics queue, and all of a family of their
        // own for compute and optical flow.
        info.graphicsQueueIndex = 1;
        info.computeQueueFamily = frame_gen.compute_family;
        info.computeQueueIndex = 0;
        if (frame_gen.native_optical_flow) {
            info.opticalFlowQueueFamily = frame_gen.optical_flow_family;
            info.opticalFlowQueueIndex = 0;
            info.useNativeOpticalFlowMode = true;
        }
    }
    if (const sl::Result result = state.api.set_vulkan_info(info); result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "Streamline did not take the Vulkan device (result {})",
                  static_cast<int>(result));
        Stop();
        return false;
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physical;
    if (const sl::Result result = state.api.is_feature_supported(sl::kFeatureDLSS, adapter);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA DLSS is not supported on this GPU or driver (Streamline result {}); "
                    "using FSR 3.1",
                    static_cast<int>(result));
        Stop();
        return false;
    }
    // Manual hooking: Streamline does its per-frame work in these.
    const auto hook = [&](auto& function, const char* name) {
        if (const FARPROC proxy = GetProcAddress(state.module, name)) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(proxy);
        } else {
            LOG_WARNING(Render_Vulkan, "sl.interposer.dll has no {}", name);
        }
    };
    hook(dld.vkCreateSwapchainKHR, "vkCreateSwapchainKHR");
    hook(dld.vkDestroySwapchainKHR, "vkDestroySwapchainKHR");
    hook(dld.vkGetSwapchainImagesKHR, "vkGetSwapchainImagesKHR");
    hook(dld.vkAcquireNextImageKHR, "vkAcquireNextImageKHR");
    hook(dld.vkQueuePresentKHR, "vkQueuePresentKHR");
    hook(dld.vkDeviceWaitIdle, "vkDeviceWaitIdle");
    state.attached = true;
    LOG_INFO(Render_Vulkan, "NVIDIA DLSS is available through Streamline");
    if (frame_gen.enabled && AttachFrameGen(state, physical)) {
        FrameGenState& fg = state.frame_gen;
        fg.sleep_worker = std::make_unique<Common::ThreadWorker>(1, "ReflexSleep");
        fg.active = true;
        LOG_INFO(Render_Vulkan, "DLSS frame generation is available ({} optical flow)",
                 frame_gen.native_optical_flow ? "native" : "interop");
    }
    return true;
}

bool IsFrameGenActive() {
    const FrameGenState& fg = GetState().frame_gen;
    return fg.active;
}

bool RecreateSurface(const vk::Instance& instance, vk::SurfaceKHR& surface, void* window) {
    State& state = GetState();
    if (!state.frame_gen.active) {
        return false;
    }
    const auto create = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        GetProcAddress(state.module, "vkCreateWin32SurfaceKHR"));
    if (!create) {
        FrameGenFailed("vkCreateWin32SurfaceKHR", 0);
        return false;
    }
    surface = vk::SurfaceKHR{}; // a window can only have one surface in use
    const VkWin32SurfaceCreateInfoKHR ci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, nullptr, 0,
                                         nullptr, static_cast<HWND>(window)};
    VkSurfaceKHR handle{};
    if (const VkResult result = create(*instance, &ci, nullptr, &handle); result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "Streamline could not create the window's surface");
        vk::Check(result);
    }
    surface = vk::SurfaceKHR(handle, *instance, instance.Dispatch());
    return true;
}

sl::FrameToken* BeginEvaluation() {
    State& state = GetState();
    FrameGenState& fg = state.frame_gen;
    if (!fg.active) {
        return nullptr;
    }
    if (fg.open_frame && fg.evaluated) {
        // Two evaluations before a present: the first frame never reaches the screen.
        Mark(sl::PCLMarker::eRenderSubmitEnd, *fg.open_frame);
        fg.open_frame = nullptr;
    }
    if (!fg.open_frame) {
        OpenFrame(state);
    }
    if (!fg.open_frame) {
        return nullptr;
    }
    Mark(sl::PCLMarker::eSimulationEnd, *fg.open_frame);
    Mark(sl::PCLMarker::eRenderSubmitStart, *fg.open_frame);
    fg.evaluated = true;
    return fg.open_frame;
}

bool FrameHasInputs() {
    return GetState().frame_gen.evaluated;
}

PresentedFrame FinishFrame() {
    State& state = GetState();
    FrameGenState& fg = state.frame_gen;
    if (!fg.active) {
        return {};
    }
    if (!fg.open_frame) {
        OpenFrame(state);
    }
    if (!fg.open_frame) {
        return {};
    }
    if (!fg.evaluated) {
        Mark(sl::PCLMarker::eSimulationEnd, *fg.open_frame);
        Mark(sl::PCLMarker::eRenderSubmitStart, *fg.open_frame);
    }
    Mark(sl::PCLMarker::eRenderSubmitEnd, *fg.open_frame);
    const PresentedFrame finished{
        .token = fg.open_frame,
        .generate = fg.evaluated,
    };
    OpenFrame(state);
    return finished;
}

void PresentStart(const PresentedFrame& frame) {
    State& state = GetState();
    FrameGenState& fg = state.frame_gen;
    if (!fg.active || !frame.token) {
        return;
    }
    if (!fg.reflex_on) {
        // Frame generation refuses to run without Reflex. Set once a swapchain exists, as this
        // applies to the swapchain.
        sl::ReflexOptions options{};
        options.mode = sl::ReflexMode::eLowLatency;
        const sl::Result result = fg.reflex_options(options);
        fg.reflex_on = true;
        if (result != sl::Result::eOk) {
            LOG_WARNING(Render_Vulkan, "slReflexSetOptions returned {}", static_cast<int>(result));
        }
    }
    fg.dlss_streak = frame.generate ? fg.dlss_streak + 1 : 0;
    const bool failed = fg.failed.load(std::memory_order_relaxed);
    const bool wanted = !failed && frame.generate &&
                        (fg.mode_on || fg.dlss_streak >= FRAME_GEN_ON_AFTER);
    if (wanted != fg.mode_on) {
        SetFrameGenMode(state, wanted);
    }
    Mark(sl::PCLMarker::ePresentStart, *static_cast<const sl::FrameToken*>(frame.token));
    if (fg.mode_on) {
        ReportFrameGen(state);
    }
}

void PresentEnd(const PresentedFrame& frame) {
    FrameGenState& fg = GetState().frame_gen;
    if (!fg.active || !frame.token) {
        return;
    }
    const auto* const token = static_cast<const sl::FrameToken*>(frame.token);
    Mark(sl::PCLMarker::ePresentEnd, *token);
    if (!fg.sleeping.exchange(true, std::memory_order_acquire)) {
        fg.sleep_worker->QueueWork([&fg, token] {
            fg.reflex_sleep(*token);
            fg.sleeping.store(false, std::memory_order_release);
        });
    }
}

void PauseFrameGen() {
    State& state = GetState();
    FrameGenState& fg = state.frame_gen;
    if (!fg.active) {
        return;
    }
    if (fg.mode_on) {
        SetFrameGenMode(state, false);
    }
    fg.dlss_streak = 0;
    fg.reflex_on = false; // the new swapchain needs Reflex set again
}

void FrameGenFailed(const char* what, int result) {
    if (!GetState().frame_gen.failed.exchange(true)) {
        LOG_ERROR(Render_Vulkan, "DLSS frame generation stopped: {} failed (result {})", what,
                  result);
    }
}

bool IsDlssAvailable() {
    const State& state = GetState();
    return state.started && state.attached;
}

void Stop() {
    State& state = GetState();
    if (!state.started) {
        return;
    }
    state.frame_gen.active = false;
    state.frame_gen.sleep_worker.reset();
    const sl::Result result = state.api.shutdown();
    state.started = false;
    state.attached = false;
    LOG_INFO(Render_Vulkan, "NVIDIA Streamline stopped (result {})", static_cast<int>(result));
}

const Api& GetApi() {
    return GetState().api;
}

} // namespace Vulkan::Streamline
