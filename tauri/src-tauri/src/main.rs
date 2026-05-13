// MeetMind Tauri shell — transparent, always-on-top overlay subscribed
// to the local SSE bus. Architecture: see
// `_shared/architecture/03_MEETMIND.md` §2.7.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    meetmind_lib::run();
}
