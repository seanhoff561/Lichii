// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include <algorithm>
#include <cstdlib>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Backend::SPIRV {
namespace {
// SPV_KHR_cooperative_matrix, which Sirit does not know.
constexpr u32 OpTypeCooperativeMatrixKHR = 4456;
constexpr u32 OpCooperativeMatrixLoadKHR = 4457;
constexpr u32 OpCooperativeMatrixStoreKHR = 4458;
constexpr u32 OpCooperativeMatrixMulAddKHR = 4459;
constexpr u32 CapabilityCooperativeMatrixKHR = 6022;
constexpr u32 ScratchWordsPerSubgroup = 256; // 1 KiB: A 128 words, B 64, C and D 64
constexpr u32 ScratchB = 128;
constexpr u32 ScratchC = 192;

std::array<bool, 2> MatrixDepthsUsed(const IR::Program& program) {
    std::array<bool, 2> needed{};
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::TensorMulAdd || inst.Flags<u32>() == 0) continue;
            const u32 depth = inst.Flags<u32>();
            if (depth != 8 && depth != 16) throw LogicError("Invalid software HMMA depth");
            needed[depth == 16] = true;
        }
    }
    return needed;
}
} // Anonymous namespace

void DefineCooperativeMatrix(EmitContext& ctx, const IR::Program& program) {
    const std::array<bool, 2> needed = MatrixDepthsUsed(program);
    if (!needed[0] && !needed[1]) {
        return;
    }
    const std::array<bool, 2> supported{ctx.profile.support_cooperative_matrix_f16_1688,
                                        ctx.profile.support_cooperative_matrix_f16_16816};
    if (!(needed[0] && supported[0]) && !(needed[1] && supported[1])) {
        return;
    }
    // Cooperative matrices take whole subgroups: every lane of the guest warp must be one of a
    // full 32-lane host subgroup, and the scratch must fit next to the program's own shared
    // memory.
    const auto& size = program.workgroup_size;
    const u32 invocations = size[0] * size[1] * size[2];
    if (program.stage != Stage::Compute || invocations == 0 || invocations % 32 != 0 ||
        !Sirit::ValidId(ctx.subgroup_local_invocation_id)) {
        return;
    }
    const u32 subgroups = invocations / 32;
    const bool block = ctx.uses_explicit_workgroup_layout && program.shared_memory_size != 0;
    const u32 offset = block ? Common::AlignUp(program.shared_memory_size, 16U) : 0;
    const u32 scratch_bytes = subgroups * ScratchWordsPerSubgroup * 4;
    const u32 total = (block ? offset : program.shared_memory_size) + scratch_bytes;
    if (ctx.profile.max_compute_shared_memory != 0 &&
        total > ctx.profile.max_compute_shared_memory) {
        return;
    }
    ctx.cooperative_matrix = {needed[0] && supported[0], needed[1] && supported[1]};
    ctx.AddExtension("SPV_KHR_cooperative_matrix");
    ctx.AddCapability(static_cast<spv::Capability>(CapabilityCooperativeMatrixKHR));
    ctx.AddCapability(spv::Capability::Float16);
    ctx.AddCapability(spv::Capability::GroupNonUniform);
    if (!Sirit::ValidId(ctx.F16[1])) {
        ctx.F16.Define(ctx, ctx.TypeFloat(16), "f16");
    }
    ctx.cooperative_f16 = ctx.F16[1];

    // Sirit merges identical types: keep the array unlike the program's own u32 shared array,
    // whose decorations would otherwise apply to both.
    u32 length = subgroups * ScratchWordsPerSubgroup;
    if (length == Common::DivCeil(program.shared_memory_size, 4U)) {
        ++length;
    }
    const Id words = ctx.TypeArray(ctx.U32[1], ctx.Const(length));
    ctx.cooperative_word_pointer = ctx.TypePointer(spv::StorageClass::Workgroup, ctx.U32[1]);
    if (block) {
        // With explicit layout every Block in workgroup memory aliases the others from offset
        // 0, so the scratch starts after the program's shared memory.
        ctx.Decorate(words, spv::Decoration::ArrayStride, 4U);
        const Id struct_type = ctx.TypeStruct(words);
        ctx.MemberDecorate(struct_type, 0U, spv::Decoration::Offset, offset);
        ctx.Decorate(struct_type, spv::Decoration::Block);
        ctx.cooperative_scratch = ctx.AddGlobalVariable(
            ctx.TypePointer(spv::StorageClass::Workgroup, struct_type),
            spv::StorageClass::Workgroup);
        ctx.Decorate(ctx.cooperative_scratch, spv::Decoration::Aliased);
    } else {
        ctx.cooperative_scratch = ctx.AddGlobalVariable(
            ctx.TypePointer(spv::StorageClass::Workgroup, words), spv::StorageClass::Workgroup);
    }
    ctx.cooperative_scratch_block = block;
    if (ctx.profile.supported_spirv >= 0x00010400) {
        ctx.interfaces.push_back(ctx.cooperative_scratch);
    }

    ctx.subgroup_id = ctx.AddGlobalVariable(ctx.TypePointer(spv::StorageClass::Input, ctx.U32[1]),
                                            spv::StorageClass::Input);
    ctx.Decorate(ctx.subgroup_id, spv::Decoration::BuiltIn, spv::BuiltIn::SubgroupId);
    ctx.interfaces.push_back(ctx.subgroup_id);

    // The marker: an empty function whose calls PatchCooperativeMatrix replaces.
    ctx.cooperative_marker = ctx.OpFunction(ctx.void_id, spv::FunctionControlMask::DontInline,
                                            ctx.TypeFunction(ctx.void_id));
    ctx.AddLabel();
    ctx.OpReturn();
    ctx.OpFunctionEnd();
    // Constants the patch refers to.
    for (const u32 value : {0U, 1U, 2U, 3U, 4U, 8U, 16U}) {
        void(ctx.Const(value));
    }
}

void DefineSoftwareMatrixFunctions(EmitContext& ctx, const IR::Program& program) {
    std::array<bool, 2> needed = MatrixDepthsUsed(program);
    for (u32 variant = 0; variant < 2; ++variant) {
        needed[variant] = needed[variant] && !ctx.cooperative_matrix[variant];
    }
    for (u32 variant = 0; variant < 2; ++variant) {
        if (!needed[variant]) continue;
        ctx.AddCapability(spv::Capability::GroupNonUniform);
        ctx.AddCapability(spv::Capability::GroupNonUniformShuffle);
        if (!ctx.profile.SupportsSubgroupStage(program.stage) ||
            ctx.profile.has_broken_spirv_subgroup_shuffle)
            throw NotImplementedException("Software HMMA requires subgroup shuffle support");
        const Id u32ty = ctx.U32[1], f32ty = ctx.F32[1];
        const auto imm = [&](u32 v) { return ctx.Const(v); };
        const auto add = [&](Id a, Id b) { return ctx.OpIAdd(u32ty, a, b); };
        const auto and_ = [&](Id a, u32 b) { return ctx.OpBitwiseAnd(u32ty, a, imm(b)); };
        const auto unpack = [&](Id v) { return ctx.OpUnpackHalf2x16(ctx.F32[2], v); };
        const auto round_half = [&](Id value) {
            // Do not express the intermediate rounding as unpack(pack(value)): drivers may
            // eliminate that round trip. Round normal significands explicitly, ties to even;
            // half subnormals lie on the exactly representable 2^-24 grid.
            const Id bits = ctx.OpBitcast(u32ty, value);
            const Id magnitude = and_(bits, 0x7fffffffU);
            const Id tie = and_(ctx.OpShiftRightLogical(u32ty, magnitude, imm(13)), 1);
            const Id rounded = and_(add(add(magnitude, imm(0xfff)), tie), ~0x1fffU);
            const Id normal = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, rounded, imm(0x47800000)), imm(0x7f800000), rounded);
            const Id scaled = ctx.OpFMul(f32ty, ctx.OpBitcast(f32ty, magnitude), ctx.Const(16777216.0f));
            const Id subnormal = ctx.OpBitcast(u32ty,
                ctx.OpFMul(f32ty, ctx.OpRoundEven(f32ty, scaled), ctx.Const(0x1p-24f)));
            const Id finite = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, magnitude, imm(0x38800000)), normal, subnormal);
            const Id result = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, magnitude, imm(0x7f800000)), magnitude, finite);
            return ctx.OpBitcast(f32ty, ctx.OpBitwiseOr(u32ty, result, and_(bits, 0x80000000)));
        };
        const Id type = ctx.TypeFunction(ctx.U32[2], ctx.U32[4], ctx.U32[2], ctx.U32[2]);
        const Id fn = ctx.OpFunction(ctx.U32[2], spv::FunctionControlMask::DontInline, type);
        ctx.software_matrix_functions[variant] = fn;
        const Id matrix_a = ctx.OpFunctionParameter(ctx.U32[4]);
        const Id matrix_b = ctx.OpFunctionParameter(ctx.U32[2]);
        const Id matrix_c = ctx.OpFunctionParameter(ctx.U32[2]);
        ctx.AddLabel();
        const Id counter = ctx.AddLocalVariable(ctx.TypePointer(spv::StorageClass::Function, u32ty),
                                           spv::StorageClass::Function);
        std::array<Id, 4> accumulators;
        for (auto& acc : accumulators)
            acc = ctx.AddLocalVariable(ctx.TypePointer(spv::StorageClass::Function, f32ty),
                                 spv::StorageClass::Function);
        std::array<Id, 8> inputs;
        for (u32 i = 0; i < 4; ++i)
            inputs[i] = ctx.OpCompositeExtract(u32ty, matrix_a, i);
        for (u32 i = 0; i < 2; ++i) {
            inputs[4 + i] = ctx.OpCompositeExtract(u32ty, matrix_b, i);
            inputs[6 + i] = ctx.OpCompositeExtract(u32ty, matrix_c, i);
        }
        for (u32 row = 0; row < 2; ++row) {
            const Id c = unpack(inputs[6 + row]);
            for (u32 col = 0; col < 2; ++col)
                ctx.OpStore(accumulators[row * 2 + col], ctx.OpCompositeExtract(f32ty, c, col));
        }
        ctx.OpStore(counter, imm(0));
        const Id lane = ctx.OpLoad(u32ty, ctx.subgroup_local_invocation_id);
        const Id warp_base = and_(lane, ~31U);
        const Id row_base = and_(lane, 28);
        const Id col_base = ctx.OpShiftLeftLogical(u32ty, and_(lane, 3), imm(3));
        const Id header = ctx.OpLabel(), body = ctx.OpLabel(), next = ctx.OpLabel(), end = ctx.OpLabel();
        ctx.OpBranch(header);
        ctx.AddLabel(header);
        const Id pair = ctx.OpLoad(u32ty, counter);
        const Id more = ctx.OpULessThan(ctx.U1, pair, imm(variant ? 8 : 4));
        ctx.OpLoopMerge(end, next, spv::LoopControlMask::DontUnroll);
        ctx.OpBranchConditional(more, body, end);
        ctx.AddLabel(body);
        const Id upper = ctx.OpUGreaterThanEqual(ctx.U1, pair, imm(4));
        const Id component = and_(pair, 3);
        const Id a_lane = add(warp_base, add(row_base, component));
        const Id b_lane = add(warp_base, add(col_base, component));
        const Id b = ctx.OpSelect(u32ty, upper, inputs[5], inputs[4]);
        std::array<Id, 2> bs;
        for (u32 col = 0; col < 2; ++col)
            bs[col] = unpack(ctx.OpGroupNonUniformShuffle(u32ty, imm(3), b,
                                                          add(b_lane, imm(col * 4))));
        const Id round_now = ctx.OpIEqual(ctx.U1, component, imm(3));
        for (u32 row = 0; row < 2; ++row) {
            const Id a = ctx.OpSelect(u32ty, upper, inputs[row + 2], inputs[row]);
            const Id as = unpack(ctx.OpGroupNonUniformShuffle(u32ty, imm(3), a, a_lane));
            for (u32 col = 0; col < 2; ++col) {
                Id value = ctx.OpLoad(f32ty, accumulators[row * 2 + col]);
                for (u32 half = 0; half < 2; ++half) {
                    value = ctx.OpFma(f32ty, ctx.OpCompositeExtract(f32ty, as, half),
                                      ctx.OpCompositeExtract(f32ty, bs[col], half), value);
                    ctx.Decorate(value, spv::Decoration::NoContraction);
                }
                ctx.OpStore(accumulators[row * 2 + col], value);
            }
        }
        // HMMA.F16 rounds after each K=8 group, including the midpoint of K=16.
        // OpSelect evaluates both operands: keep the expensive half conversion in
        // a branch so the other three pairs do no rounding work. The counter is
        // identical in every participating lane and all shuffles precede this branch.
        const Id round_block = ctx.OpLabel(), round_merge = ctx.OpLabel();
        ctx.OpSelectionMerge(round_merge, spv::SelectionControlMask::DontFlatten);
        ctx.OpBranchConditional(round_now, round_block, round_merge);
        ctx.AddLabel(round_block);
        for (const Id accumulator : accumulators) {
            ctx.OpStore(accumulator, round_half(ctx.OpLoad(f32ty, accumulator)));
        }
        ctx.OpBranch(round_merge);
        ctx.AddLabel(round_merge);
        ctx.OpBranch(next);
        ctx.AddLabel(next);
        ctx.OpStore(counter, add(pair, imm(1)));
        ctx.OpBranch(header);
        ctx.AddLabel(end);
        std::array<Id, 2> packed;
        for (u32 row = 0; row < 2; ++row)
            packed[row] = ctx.OpPackHalf2x16(u32ty, ctx.OpCompositeConstruct(ctx.F32[2],
                ctx.OpLoad(f32ty, accumulators[row * 2]), ctx.OpLoad(f32ty, accumulators[row * 2 + 1])));
        ctx.OpReturnValue(ctx.OpCompositeConstruct(ctx.U32[2], packed[0], packed[1]));
        ctx.OpFunctionEnd();
    }
}

namespace {
// One HMMA on the tensor cores. Fragment layout (PTX mma.m16n8k16 / m16n8k8, f16x2 registers,
// g = lane / 4, t = lane % 4), the same as the software version: a0 holds A[g][2t, 2t+1],
// a1 A[g+8][...], a2 and a3 the same rows at columns 8 + 2t; b0 holds B[2t, 2t+1][g], b1 the
// rows 8 + 2t; c0 and the result's first word hold C[g][2t, 2t+1], the second C[g+8][...].
Id EmitCooperativeMulAdd(EmitContext& ctx, bool k16, Id a, Id b, Id c) {
    const Id word_type = ctx.U32[1];
    const auto imm = [&](u32 v) { return ctx.Const(v); };
    const Id lane =
        ctx.OpBitwiseAnd(word_type, ctx.OpLoad(word_type, ctx.subgroup_local_invocation_id), imm(31));
    const Id base =
        ctx.OpIMul(word_type, ctx.OpLoad(word_type, ctx.subgroup_id), imm(ScratchWordsPerSubgroup));
    const Id g = ctx.OpShiftRightLogical(word_type, lane, imm(2));
    const Id t = ctx.OpBitwiseAnd(word_type, lane, imm(3));
    const Id g8 = ctx.OpIAdd(word_type, g, imm(8));
    const auto pointer = [&](Id index) {
        return ctx.cooperative_scratch_block
                   ? ctx.OpAccessChain(ctx.cooperative_word_pointer, ctx.cooperative_scratch,
                                       imm(0), index)
                   : ctx.OpAccessChain(ctx.cooperative_word_pointer, ctx.cooperative_scratch,
                                       index);
    };
    // Word index base + region + extra + row * stride + t.
    const auto word = [&](u32 region, Id row, u32 stride, u32 extra) {
        return ctx.OpIAdd(word_type, ctx.OpIAdd(word_type, base, imm(region + extra)),
                          ctx.OpIAdd(word_type, ctx.OpIMul(word_type, row, imm(stride)), t));
    };
    const auto store = [&](Id index, Id value) { ctx.OpStore(pointer(index), value); };
    const auto part = [&](Id vector, u32 index) {
        return ctx.OpCompositeExtract(word_type, vector, index);
    };
    const u32 stride = k16 ? 8 : 4; // words per A row and per B column: K halves
    store(word(0, g, stride, 0), part(a, 0));
    store(word(0, g8, stride, 0), part(a, 1));
    if (k16) {
        store(word(0, g, stride, 4), part(a, 2));
        store(word(0, g8, stride, 4), part(a, 3));
    }
    store(word(ScratchB, g, stride, 0), part(b, 0));
    if (k16) {
        store(word(ScratchB, g, stride, 4), part(b, 1));
    }
    const Id c0 = word(ScratchC, g, 4, 0);
    const Id c1 = word(ScratchC, g8, 4, 0);
    store(c0, part(c, 0));
    store(c1, part(c, 1));
    const auto barrier = [&] {
        ctx.OpControlBarrier(imm(static_cast<u32>(spv::Scope::Subgroup)),
                             imm(static_cast<u32>(spv::Scope::Subgroup)),
                             imm(static_cast<u32>(spv::MemorySemanticsMask::WorkgroupMemory |
                                                  spv::MemorySemanticsMask::AcquireRelease)));
    };
    barrier();
    const Id pa = pointer(base);
    const Id pb = pointer(ctx.OpIAdd(word_type, base, imm(ScratchB)));
    const Id pc = pointer(ctx.OpIAdd(word_type, base, imm(ScratchC)));
    const Id call = ctx.OpFunctionCall(ctx.void_id, ctx.cooperative_marker);
    ctx.cooperative_sites.push_back({call, pa, pb, pc, k16});
    barrier();
    return ctx.OpCompositeConstruct(ctx.U32[2], ctx.OpLoad(word_type, pointer(c0)),
                                    ctx.OpLoad(word_type, pointer(c1)));
}
} // Anonymous namespace

Id EmitTensorMulAdd(EmitContext& ctx, IR::Inst* inst, Id a, Id b, Id c) {
    const u32 depth = inst->Flags<u32>();
    if (depth != 8 && depth != 16) {
        throw NotImplementedException("Unsupported matrix multiply depth");
    }
    if (ctx.cooperative_matrix[depth == 16]) {
        return EmitCooperativeMulAdd(ctx, depth == 16, a, b, c);
    }
    return ctx.OpFunctionCall(ctx.U32[2], ctx.software_matrix_functions[depth == 16], a, b, c);
}

void PatchCooperativeMatrix(std::vector<u32>& code, const EmitContext& ctx) {
    if (ctx.cooperative_sites.empty()) {
        return;
    }
    // Ids of the u32 constants DefineCooperativeMatrix made, read back from the module.
    std::array<u32, 17> constant{};
    size_t first_function = 0;
    for (size_t offset = 5; offset < code.size();) {
        const u32 word_count = code[offset] >> 16;
        const u32 opcode = code[offset] & 0xFFFFu;
        if (word_count == 0) {
            break;
        }
        if (opcode == static_cast<u32>(spv::Op::OpConstant) && word_count == 4 &&
            code[offset + 1] == ctx.U32[1].value && code[offset + 3] < constant.size() &&
            constant[code[offset + 3]] == 0) {
            constant[code[offset + 3]] = code[offset + 2];
        }
        if (opcode == static_cast<u32>(spv::Op::OpFunction)) {
            first_function = offset;
            break;
        }
        offset += word_count;
    }
    for (const u32 value : {0U, 1U, 2U, 3U, 4U, 8U, 16U}) {
        if (constant[value] == 0 || first_function == 0) {
            throw LogicError("Cooperative matrix constants not found");
        }
    }
    // code[3] is the id bound; the inserts below reallocate code, so keep it in a local.
    u32 bound = code[3];
    const auto new_id = [&] { return bound++; };
    const auto matrix_type = [&](u32 rows, u32 columns, u32 use) {
        return std::array<u32, 7>{(7U << 16) | OpTypeCooperativeMatrixKHR,
                                  new_id(),
                                  ctx.cooperative_f16.value,
                                  constant[3], // subgroup scope
                                  constant[rows],
                                  constant[columns],
                                  constant[use]};
    };
    // A is MxK, B KxN, the accumulator MxN; uses: 0 A, 1 B, 2 accumulator.
    const auto a16 = matrix_type(16, 16, 0);
    const auto b16 = matrix_type(16, 8, 1);
    const auto a8 = matrix_type(16, 8, 0);
    const auto b8 = matrix_type(8, 8, 1);
    const auto acc = matrix_type(16, 8, 2);
    std::vector<u32> types;
    for (const auto* type : {&a16, &b16, &a8, &b8, &acc}) {
        types.insert(types.end(), type->begin(), type->end());
    }
    code.insert(code.begin() + static_cast<std::ptrdiff_t>(first_function), types.begin(),
                types.end());

    const u32 row_major = constant[0];
    const u32 column_major = constant[1];
    size_t patched = 0;
    for (size_t offset = first_function + types.size(); offset < code.size();) {
        const u32 word_count = code[offset] >> 16;
        if (word_count == 0) {
            break;
        }
        const u32 opcode = code[offset] & 0xFFFFu;
        if (opcode != static_cast<u32>(spv::Op::OpFunctionCall) || word_count != 4 ||
            code[offset + 3] != ctx.cooperative_marker.value) {
            offset += word_count;
            continue;
        }
        const u32 call_id = code[offset + 2];
        const auto site = std::ranges::find_if(
            ctx.cooperative_sites, [&](const auto& entry) { return entry.call.value == call_id; });
        if (site == ctx.cooperative_sites.end()) {
            throw LogicError("Unknown cooperative matrix marker");
        }
        const u32 type_a = site->k16 ? a16[1] : a8[1];
        const u32 type_b = site->k16 ? b16[1] : b8[1];
        const u32 stride = constant[site->k16 ? 8 : 4]; // words per A row and per B column
        const u32 ma = new_id();
        const u32 mb = new_id();
        const u32 mc = new_id();
        const u32 md = new_id();
        const std::array<u32, 29> replacement{
            (6U << 16) | OpCooperativeMatrixLoadKHR, type_a, ma, site->a.value, row_major, stride,
            (6U << 16) | OpCooperativeMatrixLoadKHR, type_b, mb, site->b.value, column_major,
            stride,
            (6U << 16) | OpCooperativeMatrixLoadKHR, acc[1], mc, site->c.value, row_major,
            constant[4],
            (6U << 16) | OpCooperativeMatrixMulAddKHR, acc[1], md, ma, mb, mc,
            (5U << 16) | OpCooperativeMatrixStoreKHR, site->c.value, md, row_major, constant[4],
        };
        code.erase(code.begin() + static_cast<std::ptrdiff_t>(offset),
                   code.begin() + static_cast<std::ptrdiff_t>(offset) + 4);
        code.insert(code.begin() + static_cast<std::ptrdiff_t>(offset), replacement.begin(),
                    replacement.end());
        offset += replacement.size();
        ++patched;
    }
    if (patched != ctx.cooperative_sites.size()) {
        throw LogicError("Cooperative matrix markers: {} of {} found", patched,
                         ctx.cooperative_sites.size());
    }
    code[3] = bound;
}

} // namespace Shader::Backend::SPIRV
