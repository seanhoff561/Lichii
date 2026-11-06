// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"

namespace Vulkan {

constexpr bool CanDiscardPendingDraw(bool native, bool depth_enabled, u32 index_count,
                                    u32 vertex_count) {
    // Native draws can produce persistent resources, even with depth enabled.
    if (native) {
        return false;
    }
    // Preserve the legacy async-shader heuristic.
    return depth_enabled || (index_count > 6 && vertex_count > 6);
}

constexpr bool DefersShaderEmission(bool native, size_t pending, size_t capacity) {
    return native && pending < capacity;
}

} // namespace Vulkan
