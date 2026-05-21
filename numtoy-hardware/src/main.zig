const std = @import("std");

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

pub fn pack_bits(allocator: std.mem.Allocator, src: []const u64, bit_width: u32) ![]u8 {
    if (bit_width == 0) return &[_]u8{};
    const total_bits = src.len * bit_width;
    const total_bytes = (total_bits + 7) / 8;
    const dest = try allocator.alloc(u8, total_bytes);
    @memset(dest, 0);

    for (src, 0..) |val, i| {
        const start_bit = i * bit_width;
        var bit_idx: u64 = 0;
        while (bit_idx < bit_width) : (bit_idx += 1) {
            const current_bit = start_bit + bit_idx;
            const byte_idx = current_bit / 8;
            const bit_in_byte = @as(u3, @intCast(current_bit % 8));
            const bit_val = @as(u8, @intCast((val >> @intCast(bit_idx)) & 1));
            dest[byte_idx] |= (bit_val << bit_in_byte);
        }
    }
    return dest;
}

pub fn unpack_bits(allocator: std.mem.Allocator, src: []const u8, count: usize, bit_width: u32) ![]u64 {
    const dest = try allocator.alloc(u64, count);
    if (bit_width == 0) {
        @memset(dest, 0);
        return dest;
    }
    for (0..count) |i| {
        const start_bit = i * bit_width;
        var val: u64 = 0;
        var bit_idx: u64 = 0;
        while (bit_idx < bit_width) : (bit_idx += 1) {
            const current_bit = start_bit + bit_idx;
            const byte_idx = current_bit / 8;
            const bit_in_byte = @as(u3, @intCast(current_bit % 8));
            if (byte_idx < src.len) {
                const bit_val = (src[byte_idx] >> bit_in_byte) & 1;
                val |= (@as(u64, bit_val) << @intCast(bit_idx));
            }
        }
        dest[i] = val;
    }
    return dest;
}

// C-compatible Exports
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

test "arbitrary-width integer packing" {
    const allocator = std.testing.allocator;
    
    // Demonstrate Zig's arbitrary-width u3 and u11 integer primitives
    const u3_array = [_]u3{ 5, 2, 7, 1, 0, 4, 6, 3 };
    var u64_buf: [u3_array.len]u64 = undefined;
    for (u3_array, 0..) |val, i| {
        u64_buf[i] = @as(u64, val);
    }

    const packed_bytes = try pack_bits(allocator, &u64_buf, 3);
    defer allocator.free(packed_bytes);

    const unpacked = try unpack_bits(allocator, packed_bytes, u3_array.len, 3);
    defer allocator.free(unpacked);

    for (u3_array, 0..) |val, i| {
        try std.testing.expectEqual(@as(u64, val), unpacked[i]);
    }

    // Now demonstrate packing arbitrary-width u11 integers
    const u11_array = [_]u11{ 2047, 1024, 0, 512, 123, 999 };
    var u64_buf_11: [u11_array.len]u64 = undefined;
    for (u11_array, 0..) |val, i| {
        u64_buf_11[i] = @as(u64, val);
    }

    const packed_11 = try pack_bits(allocator, &u64_buf_11, 11);
    defer allocator.free(packed_11);

    const unpacked_11 = try unpack_bits(allocator, packed_11, u11_array.len, 11);
    defer allocator.free(unpacked_11);

    for (u11_array, 0..) |val, i| {
        try std.testing.expectEqual(@as(u64, val), unpacked_11[i]);
    }
}
