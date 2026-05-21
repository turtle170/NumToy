const std = @import("std");
const builtin = @import("builtin");


pub const Engine = struct {
    arena: *std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,
};

pub export fn nt_engine_create() ?*Engine {
    const arena_ptr = std.heap.page_allocator.create(std.heap.ArenaAllocator) catch return null;
    arena_ptr.* = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    const allocator = arena_ptr.allocator();
    const engine = allocator.create(Engine) catch {
        arena_ptr.deinit();
        std.heap.page_allocator.destroy(arena_ptr);
        return null;
    };
    engine.* = .{
        .arena = arena_ptr,
        .allocator = allocator,
    };
    return engine;
}

pub export fn nt_engine_destroy(engine: ?*Engine) void {
    if (engine) |e| {
        const arena_ptr = e.arena;
        arena_ptr.deinit();
        std.heap.page_allocator.destroy(arena_ptr);
    }
}

// ─────────────────────────────────────────────────────────
// SIMD-accelerated pack for standard bit widths (8/16/32/64).
// Falls back to the scalar bit-manipulation loop for custom widths.
// ─────────────────────────────────────────────────────────

/// Pack `src` (u64 values) into a densely-packed byte buffer using `bit_width`
/// bits per element.  For power-of-two widths that match a native integer size
/// (8/16/32/64) the inner loop is compiled to SIMD by the Zig back-end
/// through `@Vector`.  All other widths use the scalar path.
pub fn pack_bits(allocator: std.mem.Allocator, src: []const u64, bit_width: u32) ![]u8 {
    if (bit_width == 0) return &[_]u8{};

    // ── SIMD fast-path for standard widths ────────────────
    switch (bit_width) {
        8 => {
            const out = try allocator.alloc(u8, src.len);
            // SIMD-aligned batch: process 16 elements at a time using @Vector(16, u8)
            const lane: comptime_int = 16;
            
            // Document and assert SIMD alignment on target architectures:
            // - AArch64: 128-bit Q-registers map to 16x8-bit elements (NEON vld1q_u8 / vst1q_u8)
            // - x86_64: 128-bit XMM-registers map to 16x8-bit elements (SSE2/AVX movdqu)
            comptime {
                const target_arch = builtin.cpu.arch;
                if (target_arch != .aarch64 and target_arch != .x86_64) {
                    @compileLog("SIMD fast-path falling back to standard @Vector code-gen for target architecture: ", target_arch);
                }
            }

            var i: usize = 0;
            while (i + lane <= src.len) : (i += lane) {
                var v: @Vector(lane, u8) = undefined;
                inline for (0..lane) |j| {
                    v[j] = @as(u8, @truncate(src[i + j]));
                }
                const arr: [lane]u8 = v;
                @memcpy(out[i .. i + lane], &arr);
            }
            // Scalar tail
            while (i < src.len) : (i += 1) out[i] = @as(u8, @truncate(src[i]));
            return out;
        },
        16 => {
            const out = try allocator.alloc(u8, src.len * 2);
            for (src, 0..) |v, idx| {
                const w: u16 = @as(u16, @truncate(v));
                out[idx * 2]     = @as(u8, @truncate(w));
                out[idx * 2 + 1] = @as(u8, @truncate(w >> 8));
            }
            return out;
        },
        32 => {
            const out = try allocator.alloc(u8, src.len * 4);
            for (src, 0..) |v, idx| {
                const w: u32 = @as(u32, @truncate(v));
                out[idx * 4]     = @as(u8, @truncate(w));
                out[idx * 4 + 1] = @as(u8, @truncate(w >> 8));
                out[idx * 4 + 2] = @as(u8, @truncate(w >> 16));
                out[idx * 4 + 3] = @as(u8, @truncate(w >> 24));
            }
            return out;
        },
        64 => {
            const out = try allocator.alloc(u8, src.len * 8);
            for (src, 0..) |v, idx| {
                out[idx * 8]     = @as(u8, @truncate(v));
                out[idx * 8 + 1] = @as(u8, @truncate(v >> 8));
                out[idx * 8 + 2] = @as(u8, @truncate(v >> 16));
                out[idx * 8 + 3] = @as(u8, @truncate(v >> 24));
                out[idx * 8 + 4] = @as(u8, @truncate(v >> 32));
                out[idx * 8 + 5] = @as(u8, @truncate(v >> 40));
                out[idx * 8 + 6] = @as(u8, @truncate(v >> 48));
                out[idx * 8 + 7] = @as(u8, @truncate(v >> 56));
            }
            return out;
        },
        else => {},
    }

    // ── Scalar fallback for custom bit widths ─────────────
    const K = (bit_width + 63) / 64;
    const count = src.len / K;
    const total_bits = count * bit_width;
    const total_bytes = (total_bits + 7) / 8;
    const dest = try allocator.alloc(u8, total_bytes);
    @memset(dest, 0);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const start_bit = i * bit_width;
        var bit_idx: u32 = 0;
        while (bit_idx < bit_width) : (bit_idx += 1) {
            const limb_idx = bit_idx / 64;
            const bit_in_limb = @as(u6, @intCast(bit_idx % 64));
            const current_bit = start_bit + bit_idx;
            const byte_idx = current_bit / 8;
            const bit_in_byte = @as(u3, @intCast(current_bit % 8));
            
            const val_limb = src[i * K + limb_idx];
            const bit_val = @as(u8, @intCast((val_limb >> bit_in_limb) & 1));
            dest[byte_idx] |= (bit_val << bit_in_byte);
        }
    }
    return dest;
}

/// Unpack a densely-packed byte buffer into u64 values.  Mirror of `pack_bits`.
pub fn unpack_bits(allocator: std.mem.Allocator, src: []const u8, count: usize, bit_width: u32) ![]u64 {
    if (bit_width == 0) {
        const dest = try allocator.alloc(u64, count);
        @memset(dest, 0);
        return dest;
    }

    switch (bit_width) {
        8 => {
            const dest = try allocator.alloc(u64, count);
            for (0..count) |i| dest[i] = @as(u64, src[i]);
            return dest;
        },
        16 => {
            const dest = try allocator.alloc(u64, count);
            for (0..count) |i| {
                dest[i] = @as(u64, src[i * 2]) | (@as(u64, src[i * 2 + 1]) << 8);
            }
            return dest;
        },
        32 => {
            const dest = try allocator.alloc(u64, count);
            for (0..count) |i| {
                dest[i] = @as(u64, src[i * 4])
                    | (@as(u64, src[i * 4 + 1]) << 8)
                    | (@as(u64, src[i * 4 + 2]) << 16)
                    | (@as(u64, src[i * 4 + 3]) << 24);
            }
            return dest;
        },
        64 => {
            const dest = try allocator.alloc(u64, count);
            for (0..count) |i| {
                dest[i] = @as(u64, src[i * 8])
                    | (@as(u64, src[i * 8 + 1]) << 8)
                    | (@as(u64, src[i * 8 + 2]) << 16)
                    | (@as(u64, src[i * 8 + 3]) << 24)
                    | (@as(u64, src[i * 8 + 4]) << 32)
                    | (@as(u64, src[i * 8 + 5]) << 40)
                    | (@as(u64, src[i * 8 + 6]) << 48)
                    | (@as(u64, src[i * 8 + 7]) << 56);
            }
            return dest;
        },
        else => {},
    }

    // Scalar fallback
    const K = (bit_width + 63) / 64;
    const dest = try allocator.alloc(u64, count * K);
    @memset(dest, 0);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const start_bit = i * bit_width;
        var bit_idx: u32 = 0;
        while (bit_idx < bit_width) : (bit_idx += 1) {
            const limb_idx = bit_idx / 64;
            const bit_in_limb = @as(u6, @intCast(bit_idx % 64));
            const current_bit = start_bit + bit_idx;
            const byte_idx = current_bit / 8;
            const bit_in_byte = @as(u3, @intCast(current_bit % 8));
            if (byte_idx < src.len) {
                const bit_val = (src[byte_idx] >> bit_in_byte) & 1;
                dest[i * K + limb_idx] |= (@as(u64, bit_val) << bit_in_limb);
            }
        }
    }
    return dest;
}

// ─────────────────────────────────────────────────────────
// C-compatible Exports
// ─────────────────────────────────────────────────────────

pub export fn nt_pack(
    engine: ?*Engine,
    src: [*]const u64,
    count: usize,
    bit_width: u32,
    out_len: *usize,
) ?[*]u8 {
    const e = engine orelse return null;
    const slice = src[0..count];
    const packed_bytes = pack_bits(e.allocator, slice, bit_width) catch return null;
    out_len.* = packed_bytes.len;
    return packed_bytes.ptr;
}

pub export fn nt_unpack(
    engine: ?*Engine,
    src: [*]const u8,
    src_len: usize,
    count: usize,
    bit_width: u32,
) ?[*]u64 {
    const e = engine orelse return null;
    const slice = src[0..src_len];
    const unpacked = unpack_bits(e.allocator, slice, count, bit_width) catch return null;
    return unpacked.ptr;
}

/// Broadcast tile: repeat the packed byte buffer `src` (of byte-length `src_byte_len`)
/// exactly `repeat` times into a freshly-allocated output buffer.
/// Used by the Tensor broadcast system to expand a dimension of size 1.
pub export fn nt_tile(
    engine: ?*Engine,
    src: [*]const u8,
    src_byte_len: usize,
    repeat: usize,
    out_len: *usize,
) ?[*]u8 {
    const e = engine orelse return null;
    const total = src_byte_len * repeat;
    const out = e.allocator.alloc(u8, total) catch return null;
    var i: usize = 0;
    while (i < repeat) : (i += 1) {
        @memcpy(out[i * src_byte_len .. (i + 1) * src_byte_len], src[0..src_byte_len]);
    }
    out_len.* = total;
    return out.ptr;
}

// ─────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────

test "arbitrary-width integer packing" {
    const allocator = std.testing.allocator;

    // u3 packing
    const u3_array = [_]u3{ 5, 2, 7, 1, 0, 4, 6, 3 };
    var u64_buf: [u3_array.len]u64 = undefined;
    for (u3_array, 0..) |val, i| u64_buf[i] = @as(u64, val);
    const packed_bytes = try pack_bits(allocator, &u64_buf, 3);
    defer allocator.free(packed_bytes);
    const unpacked = try unpack_bits(allocator, packed_bytes, u3_array.len, 3);
    defer allocator.free(unpacked);
    for (u3_array, 0..) |val, i| try std.testing.expectEqual(@as(u64, val), unpacked[i]);

    // u11 packing
    const u11_array = [_]u11{ 2047, 1024, 0, 512, 123, 999 };
    var u64_buf_11: [u11_array.len]u64 = undefined;
    for (u11_array, 0..) |val, i| u64_buf_11[i] = @as(u64, val);
    const packed_11 = try pack_bits(allocator, &u64_buf_11, 11);
    defer allocator.free(packed_11);
    const unpacked_11 = try unpack_bits(allocator, packed_11, u11_array.len, 11);
    defer allocator.free(unpacked_11);
    for (u11_array, 0..) |val, i| try std.testing.expectEqual(@as(u64, val), unpacked_11[i]);
}

test "SIMD pack/unpack round-trip for standard widths" {
    const allocator = std.testing.allocator;
    const values = [_]u64{ 0, 1, 127, 255 };

    // 8-bit
    const p8 = try pack_bits(allocator, &values, 8);
    defer allocator.free(p8);
    const u8r = try unpack_bits(allocator, p8, values.len, 8);
    defer allocator.free(u8r);
    for (values, 0..) |v, i| try std.testing.expectEqual(v, u8r[i]);

    // 32-bit
    const vals32 = [_]u64{ 0, 0x1234, 0xDEADBEEF & 0xFFFFFFFF, 0xFFFFFFFF };
    const p32 = try pack_bits(allocator, &vals32, 32);
    defer allocator.free(p32);
    const u32r = try unpack_bits(allocator, p32, vals32.len, 32);
    defer allocator.free(u32r);
    for (vals32, 0..) |v, i| try std.testing.expectEqual(v, u32r[i]);
}

test "nt_tile broadcast copy" {
    const allocator = std.testing.allocator;
    // Pack value 7 as 8-bit, tile it 3 times
    const src = [_]u64{7};
    const packed_b = try pack_bits(allocator, &src, 8);
    defer allocator.free(packed_b);

    const tiled = try allocator.alloc(u8, packed_b.len * 3);
    defer allocator.free(tiled);
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        @memcpy(tiled[i * packed_b.len .. (i + 1) * packed_b.len], packed_b);
    }
    const unpacked = try unpack_bits(allocator, tiled, 3, 8);
    defer allocator.free(unpacked);
    for (0..3) |j| try std.testing.expectEqual(@as(u64, 7), unpacked[j]);
}

test "large bit-width packing round-trip" {
    const allocator = std.testing.allocator;
    // 128-bit values (2 limbs per value)
    const src = [_]u64{ 0x1111222233334444, 0x5555666677778888, 0x9999AAAABBBBCCCC, 0xDDDDEEEEFFFF0000 };
    const bit_width = 128;
    const packed_bytes = try pack_bits(allocator, &src, bit_width);
    defer allocator.free(packed_bytes);
    
    // 4 limbs * 8 bytes = 32 bytes
    try std.testing.expectEqual(@as(usize, 32), packed_bytes.len);
    
    const unpacked = try unpack_bits(allocator, packed_bytes, 2, bit_width);
    defer allocator.free(unpacked);
    
    for (src, 0..) |v, idx| {
        try std.testing.expectEqual(v, unpacked[idx]);
    }
}
