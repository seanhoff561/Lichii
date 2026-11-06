// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace VideoCommon::Nx2Perf {

inline bool Verify() {
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_PERF_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

} // namespace VideoCommon::Nx2Perf
