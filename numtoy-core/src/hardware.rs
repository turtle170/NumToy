use std::ffi::c_void;

#[repr(C)]
pub struct EngineOpaque {
    _unused: [u8; 0],
}

extern "C" {
    pub fn nt_engine_create() -> *mut EngineOpaque;
    pub fn nt_engine_destroy(engine: *mut EngineOpaque);
    pub fn nt_pack(
        engine: *mut EngineOpaque,
        src: *const u64,
        count: usize,
        bit_width: u32,
        out_len: *mut usize,
    ) -> *mut u8;
    pub fn nt_unpack(
        engine: *mut EngineOpaque,
        src: *const u8,
        src_len: usize,
        count: usize,
        bit_width: u32,
    ) -> *mut u64;
}

pub struct HardwareEngine {
    ptr: *mut EngineOpaque,
}

impl HardwareEngine {
    pub fn new() -> Self {
        let ptr = unsafe { nt_engine_create() };
        assert!(!ptr.is_null(), "Failed to create Zig Engine");
        Self { ptr }
    }

    pub fn pack(&self, src: &[u64], bit_width: u32) -> Vec<u8> {
        if src.is_empty() || bit_width == 0 {
            return Vec::new();
        }
        let mut out_len = 0;
        let ptr = unsafe {
            nt_pack(
                self.ptr,
                src.as_ptr(),
                src.len(),
                bit_width,
                &mut out_len as *mut usize,
            )
        };
        if ptr.is_null() {
            return Vec::new();
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, out_len) };
        slice.to_vec()
    }

    pub fn unpack(&self, src: &[u8], count: usize, bit_width: u32) -> Vec<u64> {
        if count == 0 || bit_width == 0 {
            return vec![0; count];
        }
        let ptr = unsafe {
            nt_unpack(
                self.ptr,
                src.as_ptr(),
                src.len(),
                count,
                bit_width,
            )
        };
        if ptr.is_null() {
            return vec![0; count];
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.to_vec()
    }
}

impl Drop for HardwareEngine {
    fn drop(&mut self) {
        unsafe {
            nt_engine_destroy(self.ptr);
        }
    }
}

unsafe impl Send for HardwareEngine {}
unsafe impl Sync for HardwareEngine {}
