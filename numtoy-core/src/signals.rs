//! OS-level signal and exception trapping for NumToy.
//!
//! Call `signals::install()` once at startup (done automatically by the
//! Python binding's module-init function).  Handlers are registered for:
//!
//! | Signal / Event          | Platform       | Action                                      |
//! |-------------------------|----------------|---------------------------------------------|
//! | SIGINT  / CTRL_C_EVENT  | all            | Clear Xtreme mode, re-raise for clean exit  |
//! | SIGSEGV / ACCESS_VIOLATION | all         | Clear Xtreme mode, re-raise for core dump   |
//!
//! On Windows the process priority is also reset to NORMAL before re-raising
//! so that the OS can schedule competing cleanup threads.

use std::sync::atomic::Ordering;

// ─── Shared cleanup ───────────────────────────────────────────────────────────

/// Executed in every signal handler before re-raising.
/// Cheap enough to call from an async-signal context (only atomic stores).
#[inline(always)]
fn emergency_cleanup() {
    // Break the Xtreme hot-spin immediately.
    crate::pool::XTREME_ACTIVE.store(false, Ordering::SeqCst);
    // Drop back to Default mode so worker threads stop TIME_CRITICAL spin.
    crate::GLOBAL_EXECUTION_MODE.store(0 /* Default */, Ordering::SeqCst);
}

// ─── Windows ─────────────────────────────────────────────────────────────────

#[cfg(windows)]
mod platform {
    use super::emergency_cleanup;
    use windows::Win32::System::Threading::{
        GetCurrentProcess, SetPriorityClass, NORMAL_PRIORITY_CLASS,
    };
    use windows::Win32::System::Console::{SetConsoleCtrlHandler, CTRL_C_EVENT};
    use windows::core::BOOL;

    // PHANDLER_ROUTINE = Option<unsafe extern "system" fn(u32) -> windows::core::BOOL>
    // Return TRUE (non-zero) to indicate handled; FALSE to pass to next handler.
    unsafe extern "system" fn ctrl_handler(ctrl_type: u32) -> BOOL {
        if ctrl_type == CTRL_C_EVENT {
            emergency_cleanup();
            unsafe {
                let _ = SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
            }
        }
        // Return FALSE (0) so the next handler (default) also runs → process exits.
        BOOL(0)
    }

    pub fn install() {
        unsafe {
            // Ctrl+C / Ctrl+Break — pass `true` to add this handler
            let _ = SetConsoleCtrlHandler(Some(ctrl_handler), true);
        }
    }
}

// ─── Linux / macOS ───────────────────────────────────────────────────────────

#[cfg(not(windows))]
mod platform {
    use super::emergency_cleanup;
    use std::sync::atomic::{AtomicBool, Ordering};

    static HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

    extern "C" fn sigint_handler(_sig: libc::c_int) {
        emergency_cleanup();
        // Re-raise with default handler so the shell sees Ctrl+C properly.
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::raise(libc::SIGINT);
        }
    }

    extern "C" fn sigsegv_handler(_sig: libc::c_int) {
        emergency_cleanup();
        unsafe {
            libc::signal(libc::SIGSEGV, libc::SIG_DFL);
            libc::raise(libc::SIGSEGV);
        }
    }

    pub fn install() {
        if HANDLER_INSTALLED
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return; // already installed
        }
        unsafe {
            libc::signal(libc::SIGINT,  sigint_handler  as libc::sighandler_t);
            libc::signal(libc::SIGSEGV, sigsegv_handler as libc::sighandler_t);
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Register all NumToy signal handlers.  Safe to call multiple times (idempotent).
pub fn install() {
    platform::install();
}
