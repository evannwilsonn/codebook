//! A tiny C-style interface for the browser demo (wasm32-unknown-unknown, no bindings crate).
//!
//! The page copies a `.tiktoken` file into wasm memory with `cb_alloc`, loads it with
//! `cb_load`, then calls `cb_encode` with UTF-8 text and reads the ids from `cb_ids`.
//! Token bytes for display come from the rank file the page already has.

use std::cell::RefCell;

use crate::{Rank, Tokenizer, GPT2_PATTERN, GPT4_PATTERN};

thread_local! {
    static SLOTS: RefCell<Vec<Tokenizer>> = const { RefCell::new(Vec::new()) };
    static IDS: RefCell<Vec<Rank>> = const { RefCell::new(Vec::new()) };
}

/// Reserves `len` bytes the page can write into.
#[no_mangle]
pub extern "C" fn cb_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Frees a buffer from `cb_alloc` with the same length.
///
/// # Safety
/// `ptr` and `len` must come from one `cb_alloc` call, and the buffer must not be used again.
#[no_mangle]
pub unsafe extern "C" fn cb_free(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
}

unsafe fn text<'a>(ptr: *const u8, len: usize) -> &'a str {
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len)).unwrap_or("")
}

/// Loads a vocabulary. `pattern` is 4 for the GPT-4 split, 2 for GPT-2. Returns its slot, or -1.
///
/// # Safety
/// `ptr..ptr+len` must be initialised memory from `cb_alloc`.
#[no_mangle]
pub unsafe extern "C" fn cb_load(ptr: *const u8, len: usize, pattern: u32) -> i32 {
    let pattern = if pattern == 2 {
        GPT2_PATTERN
    } else {
        GPT4_PATTERN
    };
    match Tokenizer::from_tiktoken(text(ptr, len), pattern) {
        Ok(t) => SLOTS.with(|s| {
            let mut s = s.borrow_mut();
            s.push(t);
            (s.len() - 1) as i32
        }),
        Err(_) => -1,
    }
}

/// Encodes UTF-8 text with the vocabulary in `slot` and returns the number of ids.
///
/// # Safety
/// `ptr..ptr+len` must be initialised memory from `cb_alloc`.
#[no_mangle]
pub unsafe extern "C" fn cb_encode(slot: u32, ptr: *const u8, len: usize) -> u32 {
    let ids = SLOTS.with(|s| {
        s.borrow()
            .get(slot as usize)
            .map(|t| t.encode(text(ptr, len)))
            .unwrap_or_default()
    });
    let n = ids.len() as u32;
    IDS.with(|buf| *buf.borrow_mut() = ids);
    n
}

/// Where the ids from the last `cb_encode` are. Valid until the next call.
#[no_mangle]
pub extern "C" fn cb_ids() -> *const Rank {
    IDS.with(|buf| buf.borrow().as_ptr())
}
