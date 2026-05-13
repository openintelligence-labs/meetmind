// Tauri shell entry — minimal v0.13. Hooks for tray icon + global
// hotkeys land in v0.14.

use tauri::{
    menu::{Menu, MenuItem},
    tray::{TrayIconBuilder, TrayIconEvent},
    Manager, WindowEvent,
};

#[tauri::command]
fn meetmind_endpoint() -> String {
    // Default SSE endpoint — the overlay can override via the controls.
    "http://127.0.0.1:7857".to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![meetmind_endpoint])
        .setup(|app| {
            // Tray icon — clicking toggles the overlay.
            let toggle = MenuItem::with_id(app, "toggle", "Show / hide overlay", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit MeetMind", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&toggle, &quit])?;
            let _tray = TrayIconBuilder::with_id("meetmind-tray")
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "toggle" => {
                        if let Some(w) = app.get_webview_window("overlay") {
                            if w.is_visible().unwrap_or(false) {
                                let _ = w.hide();
                            } else {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { .. } = event {
                        if let Some(w) = tray.app_handle().get_webview_window("overlay") {
                            let _ = if w.is_visible().unwrap_or(false) { w.hide() } else { w.show() };
                        }
                    }
                })
                .build(app)?;

            // Hide instead of close on red-button click.
            if let Some(w) = app.get_webview_window("overlay") {
                let win = w.clone();
                w.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = win.hide();
                    }
                });
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
