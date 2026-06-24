//! nanorts — a tiny hand-crafted real-time strategy game.
//!
//! RTS in a teacup: mine minerals, build a barracks, raise an army, and
//! crush the red base before its attacks crush yours. Everything is drawn into a
//! software framebuffer; the only dependency is a window to blit it into.
//!
//! Controls:
//!   Left-drag      select units      Right-click   move / attack / gather
//!   W (base)       train worker      E (barracks)  train soldier
//!   B (worker)     place barracks    A             attack-move
//!   S              stop              H             toggle help
//!   Arrows / edges scroll            Minimap       click to jump
//!   Esc            cancel / deselect R             restart (when game over)
//!
//! The game opens to a main menu (Single Player / Host / Join / Quit); pause and
//! the end screen route back to it. Matches are 2–4 faction free-for-alls: solo
//! vs up to 3 AI, or up to 4 humans online. Multiplayer is serverless
//! deterministic lockstep — no server, the host relays commands in a star, AI
//! factions are simulated locally — set up from the menu, or via CLI shortcuts:
//!   nanorts --host [port]   host a match + beacon on the LAN  (default :7777)
//!   nanorts --join          auto-discover and join a LAN host
//!   nanorts --join <addr>   join a specific host, e.g. 192.168.0.10:7777

mod ai;
mod audio;
mod font;
mod gfx;
mod net;
mod vec;
mod world;

use gfx::{rgb, Canvas};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use vec::{v2, V2};
use world::{cost, radius, Kind, Order, Team, World};

const HUD_H: i32 = 30;
const MM_W: i32 = 220;
const MM_H: i32 = 138;

// We render the game into a fixed-height "design" buffer (this many world units
// tall, scaled to fill the screen) and blit it up to the real window. This
// keeps the on-screen view a sensible zoom on any resolution and avoids
// minifb's (crash-prone) scaler.
const DESIGN_H: i32 = 720;
fn design_width(sw: i32, sh: i32) -> i32 {
    if sh <= 0 {
        return 1280;
    }
    (DESIGN_H * sw / sh).max(640) // match the screen aspect, no letterboxing
}
fn build_x_map(screen_w: i32, design_w: i32) -> Vec<u32> {
    (0..screen_w.max(1))
        .map(|x| ((x * design_w) / screen_w.max(1)) as u32)
        .collect()
}
/// Nearest-neighbour upscale of the design buffer into the full-screen buffer.
/// Hot path at high resolutions, so it works on row slices and iterates to keep
/// the inner loop free of bounds checks.
fn blit_scaled(design: &Canvas, screen: &mut Canvas, x_map: &[u32]) {
    let dgw = design.w as usize;
    let dgh = design.h as usize;
    let scw = screen.w as usize;
    let sch = screen.h as usize;
    if dgw == 0 || dgh == 0 || x_map.len() < scw {
        return;
    }
    let xm = &x_map[..scw];
    for y in 0..sch {
        let sy = (y * dgh) / sch;
        let src = &design.buf[sy * dgw..sy * dgw + dgw];
        let dst = &mut screen.buf[y * scw..y * scw + scw];
        for (d, &x) in dst.iter_mut().zip(xm.iter()) {
            *d = src[x as usize];
        }
    }
}

/// Keep the design/screen buffers matched to the live window size. Shared by
/// the lobby screens and (conceptually) the main loop's own resize check.
fn sync_canvas(window: &Window, screen: &mut Canvas, canvas: &mut Canvas, x_map: &mut Vec<u32>) {
    let (cw, ch) = window.get_size();
    if cw > 0 && ch > 0 && (cw as i32 != screen.w || ch as i32 != screen.h) {
        *screen = Canvas::new(cw as i32, ch as i32);
        *canvas = Canvas::new(design_width(cw as i32, ch as i32), DESIGN_H);
        *x_map = build_x_map(screen.w, canvas.w);
    }
}

/// Upscale + present the design canvas, skipping the frame on a size mismatch
/// (e.g. mid-resize or minimized) so we never trip minifb's scaler.
fn present_frame(window: &mut Window, screen: &mut Canvas, canvas: &Canvas, x_map: &[u32]) {
    blit_scaled(canvas, screen, x_map);
    let (cw, ch) = window.get_size();
    if cw == screen.w as usize && ch == screen.h as usize && cw > 0 {
        let _ = window.update_with_buffer(&screen.buf, cw, ch);
    } else {
        window.update();
    }
}

/// The pre-game connection screen: a dark ops-room panel with a sweeping radar
/// beam, a big status line, and a sub-line (address / hint). `accent` tints the
/// beam and frame; `t` drives the animation.
fn draw_lobby(c: &mut Canvas, t: u64, heading: &str, big: &str, sub: &str, hint: &str, accent: u32) {
    let (w, h) = (c.w, c.h);
    c.clear(rgb(6, 8, 12));
    // Faint scanline grid for the "tactical display" feel.
    let mut y = (t as i32 / 2) % 26;
    while y < h {
        c.fill_rect_a(0, y, w, 1, rgb(40, 80, 70), 0.05);
        y += 26;
    }
    // Sweeping radar beam from the centre.
    let (cx, cy) = (w / 2, h / 2);
    let ang = (t as f32) * 0.045;
    let reach = (w.max(h)) as f32;
    for i in 0..26 {
        let a = ang - i as f32 * 0.03;
        let fade = 1.0 - i as f32 / 26.0;
        let ex = cx + (a.cos() * reach) as i32;
        let ey = cy + (a.sin() * reach * 0.62) as i32;
        let col = gfx::mix(rgb(6, 8, 12), accent, 0.10 + 0.22 * fade);
        c.line(cx, cy, ex, ey, col);
    }
    // Concentric range rings.
    for r in [70, 140, 210, 280] {
        c.circle(cx, cy, r, gfx::mix(rgb(6, 8, 12), accent, 0.16));
    }
    // Pulsing core.
    let pulse = ((t as f32 * 0.08).sin() * 0.5 + 0.5) * 0.8 + 0.2;
    c.fill_circle_add(cx, cy, 5, accent, pulse);

    // Text stack, centred.
    c.text_center(cx, cy - 150, heading, gfx::mix(accent, rgb(255, 255, 255), 0.4), 2);
    c.text_center(cx, cy - 110, big, rgb(245, 248, 250), 5);
    if !sub.is_empty() {
        c.text_center(cx, cy - 56, sub, gfx::mix(accent, rgb(230, 240, 240), 0.5), 3);
    }
    if !hint.is_empty() {
        c.text_center(cx, h - 70, hint, rgb(150, 165, 170), 2);
    }
    // Outer frame.
    c.rect(18, 18, w - 36, h - 36, gfx::mix(rgb(6, 8, 12), accent, 0.5));
    c.rect(20, 20, w - 40, h - 40, gfx::mix(rgb(6, 8, 12), accent, 0.25));
}

/// The live multiplayer status pill at the top of the screen: a green
/// "LOCKSTEP OK" with the current frame and last-verified-sync step, a red
/// desync alarm if the two sims ever diverge, or a disconnect warning.
fn draw_sync_badge(c: &mut Canvas, ls: &net::Lockstep) {
    let cx = c.w / 2;
    let y = HUD_H + 6; // just below the HUD bar, so it never overlaps the readouts
    let pill = |c: &mut Canvas, txt: &str, fg: u32, bg: u32, glow: f32| {
        let bw = Canvas::text_width(txt, 2) + 18;
        c.fill_rect_a(cx - bw / 2, y, bw, 20, bg, 0.6);
        c.rect(cx - bw / 2, y, bw, 20, gfx::mix(bg, fg, glow));
        c.text_center(cx, y + 4, txt, fg, 2);
    };
    if ls.disconnected {
        pill(c, "OPPONENT DISCONNECTED", rgb(255, 90, 80), rgb(40, 4, 4), 0.7);
    } else if let Some(s) = ls.desync_step {
        // Glitchy alarm — jitter the colour so it reads as "alarm".
        let flick = ((ls.sim_step / 4) % 2) as f32;
        let fg = gfx::mix(rgb(255, 70, 60), rgb(255, 200, 60), flick);
        pill(c, &format!("! DESYNC AT STEP {} !", s), fg, rgb(50, 4, 4), 0.9);
    } else {
        let txt = format!("LOCKSTEP OK   F{}   SYNC {}", ls.sim_step, ls.last_synced);
        pill(c, &txt, rgb(120, 230, 150), rgb(2, 20, 14), 0.5);
    }
}

// Minimap sits in the bottom-right corner of whatever resolution we run at.
fn mm_x(screen_w: i32) -> i32 {
    screen_w - MM_W - 10
}
fn mm_y(screen_h: i32) -> i32 {
    // Sit clear of the 22px action bar pinned to the bottom edge (its frame
    // extends 4px below this, leaving an ~8px gap above the bar).
    screen_h - MM_H - 34
}

struct Ui {
    drag_start: Option<V2>,
    dragging: bool,
    build_mode: Option<Kind>,
    attack_pending: bool,
    show_help: bool,
    menu: bool, // pause menu (resume / surrender / quit)
}

fn seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}

/// Best-effort full-screen target: the primary monitor's geometry as
/// `(width, height, x, y)`, falling back to a plain 1280x720 window if unknown.
///
/// Linux parses `xrandr` (X11/XWayland) so we fill one monitor rather than a
/// multi-head span; Windows asks Win32 for the primary monitor; any other
/// platform takes the windowed fallback.
#[cfg(target_os = "linux")]
fn detect_screen() -> (usize, usize, i32, i32) {
    if let Ok(out) = std::process::Command::new("xrandr").arg("--query").output() {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            let mut first: Option<(usize, usize, i32, i32)> = None;
            let mut primary: Option<(usize, usize, i32, i32)> = None;
            for line in text.lines() {
                if !line.contains(" connected") {
                    continue;
                }
                let geom = line.split_whitespace().find_map(parse_geom);
                if let Some(g) = geom {
                    if line.contains("primary") {
                        primary = Some(g);
                    }
                    if first.is_none() {
                        first = Some(g);
                    }
                }
            }
            if let Some(g) = primary.or(first) {
                return g;
            }
        }
    }
    (1280, 720, 0, 0) // windowed fallback
}

#[cfg(target_os = "windows")]
fn detect_screen() -> (usize, usize, i32, i32) {
    // Primary monitor size straight from Win32 — no windowing crate needed, just
    // a tiny FFI call. SM_CXSCREEN = 0, SM_CYSCREEN = 1.
    #[link(name = "user32")]
    extern "system" {
        fn GetSystemMetrics(index: i32) -> i32;
    }
    let (w, h) = unsafe { (GetSystemMetrics(0), GetSystemMetrics(1)) };
    if w > 0 && h > 0 {
        (w as usize, h as usize, 0, 0)
    } else {
        (1280, 720, 0, 0)
    }
}

#[cfg(target_os = "macos")]
fn detect_screen() -> (usize, usize, i32, i32) {
    // Main display size (in points) from CoreGraphics — no extra crate, just FFI.
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGMainDisplayID() -> u32;
        fn CGDisplayPixelsWide(display: u32) -> usize;
        fn CGDisplayPixelsHigh(display: u32) -> usize;
    }
    let (w, h) = unsafe {
        let d = CGMainDisplayID();
        (CGDisplayPixelsWide(d), CGDisplayPixelsHigh(d))
    };
    if w > 0 && h > 0 {
        (w, h, 0, 0)
    } else {
        (1280, 720, 0, 0)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn detect_screen() -> (usize, usize, i32, i32) {
    (1280, 720, 0, 0) // windowed fallback elsewhere
}

/// Parse an xrandr geometry token `WxH+X+Y` into `(w, h, x, y)`.
#[cfg(target_os = "linux")]
fn parse_geom(tok: &str) -> Option<(usize, usize, i32, i32)> {
    let (wh, xy) = tok.split_once('+')?;
    let (w, h) = wh.split_once('x')?;
    let (x, y) = xy.split_once('+')?;
    let w = w.parse::<usize>().ok()?;
    let h = h.parse::<usize>().ok()?;
    let x = x.parse::<i32>().ok()?;
    let y = y.parse::<i32>().ok()?;
    (w >= 640 && h >= 480).then_some((w, h, x, y))
}

// ---- front-end: menus & multiplayer setup ---------------------------------

const MP_PORT: u16 = 7777;

/// The app's top-level state, threaded through the main loop.
enum AppNext {
    Menu,
    Single(usize), // free-for-all with this many factions (1 human + AI)
    Host(u16),
    Join(String),
    Quit,
}

/// How a match ended, from the game loop's point of view.
enum GameOutcome {
    ToMenu,
    Quit,
}

/// A click action shared by the pause and game-over screens.
#[derive(Clone, Copy)]
enum MenuAct {
    Resume,
    Surrender,
    Restart,
    ToMenu,
    Quit,
}

/// A clickable button: a labelled rect with hover feedback.
struct Btn {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    label: String,
    accent: u32,
}
impl Btn {
    fn new(cx: i32, y: i32, w: i32, h: i32, label: &str, accent: u32) -> Btn {
        Btn { x: cx - w / 2, y, w, h, label: label.to_string(), accent }
    }
    fn hit(&self, m: V2) -> bool {
        m.x >= self.x as f32
            && m.x <= (self.x + self.w) as f32
            && m.y >= self.y as f32
            && m.y <= (self.y + self.h) as f32
    }
    fn draw(&self, c: &mut Canvas, hover: bool) {
        let base = rgb(10, 14, 12);
        let bg = if hover { gfx::mix(base, self.accent, 0.30) } else { base };
        c.fill_rect_a(self.x, self.y, self.w, self.h, bg, 0.92);
        let border = if hover { self.accent } else { gfx::mix(base, self.accent, 0.55) };
        c.rect(self.x, self.y, self.w, self.h, border);
        let col = if hover { rgb(248, 252, 252) } else { rgb(204, 216, 216) };
        c.text_center(self.x + self.w / 2, self.y + self.h / 2 - 7, &self.label, col, 3);
    }
}

/// The shared ops-room backdrop behind every front-end screen.
fn draw_menu_bg(c: &mut Canvas, t: u64, title: &str, subtitle: &str, accent: u32) {
    let (w, h) = (c.w, c.h);
    c.clear(rgb(6, 8, 12));
    let mut y = (t as i32 / 2) % 26;
    while y < h {
        c.fill_rect_a(0, y, w, 1, rgb(40, 80, 70), 0.05);
        y += 26;
    }
    let (cx, cy) = (w / 2, h / 2);
    let ang = (t as f32) * 0.03;
    let reach = w.max(h) as f32;
    for i in 0..28 {
        let a = ang - i as f32 * 0.028;
        let fade = 1.0 - i as f32 / 28.0;
        let ex = cx + (a.cos() * reach) as i32;
        let ey = cy + (a.sin() * reach * 0.6) as i32;
        c.line(cx, cy, ex, ey, gfx::mix(rgb(6, 8, 12), accent, 0.05 + 0.13 * fade));
    }
    c.text_center(cx, 78, title, gfx::mix(accent, rgb(255, 255, 255), 0.5), 9);
    if !subtitle.is_empty() {
        c.text_center(cx, 156, subtitle, gfx::mix(accent, rgb(230, 240, 240), 0.5), 2);
    }
    c.rect(18, 18, w - 36, h - 36, gfx::mix(rgb(6, 8, 12), accent, 0.4));
}

/// Map a key to the character it types in the address field (digits, dot, colon).
fn key_char(k: Key) -> Option<char> {
    Some(match k {
        Key::Key0 | Key::NumPad0 => '0',
        Key::Key1 | Key::NumPad1 => '1',
        Key::Key2 | Key::NumPad2 => '2',
        Key::Key3 | Key::NumPad3 => '3',
        Key::Key4 | Key::NumPad4 => '4',
        Key::Key5 | Key::NumPad5 => '5',
        Key::Key6 | Key::NumPad6 => '6',
        Key::Key7 | Key::NumPad7 => '7',
        Key::Key8 | Key::NumPad8 => '8',
        Key::Key9 | Key::NumPad9 => '9',
        Key::Period | Key::NumPadDot => '.',
        Key::Semicolon => ':',
        _ => return None,
    })
}

/// Default the port if the typed address has none.
fn normalize_addr(s: &str) -> String {
    if s.contains(':') {
        s.to_string()
    } else {
        format!("{}:{}", s, MP_PORT)
    }
}

/// The main menu: Single Player / Host / Join / Quit. Returns the chosen state.
fn run_main_menu(window: &mut Window, screen: &mut Canvas, canvas: &mut Canvas, x_map: &mut Vec<u32>, windowed: bool) -> AppNext {
    let mut t = 0u64;
    let mut last_left = window.get_mouse_down(MouseButton::Left);
    let mut last_mouse = v2(canvas.w as f32 / 2.0, canvas.h as f32 / 2.0);
    let accent = rgb(90, 200, 255);
    loop {
        if !window.is_open() {
            return AppNext::Quit;
        }
        sync_canvas(window, screen, canvas, x_map);
        if let Some((rx, ry)) = window.get_mouse_pos(MouseMode::Discard) {
            last_mouse = v2(rx * canvas.w as f32 / screen.w.max(1) as f32, ry * canvas.h as f32 / screen.h.max(1) as f32);
        }
        let mouse = last_mouse;
        let left = window.get_mouse_down(MouseButton::Left);
        let click = left && !last_left;
        last_left = left;

        draw_menu_bg(canvas, t, "NANORTS", "RTS IN A TEACUP", accent);
        let cx = canvas.w / 2;
        let labels = ["SINGLE PLAYER", "HOST GAME (LAN)", "JOIN GAME (LAN)", "QUIT"];
        let accents = [rgb(120, 230, 140), rgb(90, 200, 255), rgb(255, 200, 90), rgb(235, 120, 110)];
        let (bw, bh, gap) = (380, 56, 18);
        let y0 = canvas.h / 2 - 70;
        let mut chosen: Option<usize> = None;
        for (i, lab) in labels.iter().enumerate() {
            let b = Btn::new(cx, y0 + i as i32 * (bh + gap), bw, bh, lab, accents[i]);
            let hover = b.hit(mouse);
            b.draw(canvas, hover);
            if click && hover {
                chosen = Some(i);
            }
        }
        canvas.text_center(cx, canvas.h - 54, "CLICK  OR  PRESS 1-4", rgb(120, 135, 140), 2);
        for k in window.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Key1 | Key::NumPad1 => chosen = Some(0),
                Key::Key2 | Key::NumPad2 => chosen = Some(1),
                Key::Key3 | Key::NumPad3 => chosen = Some(2),
                Key::Key4 | Key::NumPad4 | Key::Escape | Key::Q => chosen = Some(3),
                _ => {}
            }
        }
        present_frame(window, screen, canvas, x_map);
        if !windowed {
            fullscreen_tick(t);
        }
        match chosen {
            Some(0) => match run_skirmish_setup(window, screen, canvas, x_map, windowed) {
                AppNext::Menu => last_left = true,
                other => return other,
            },
            Some(1) => return AppNext::Host(MP_PORT),
            Some(2) => match run_join_browse(window, screen, canvas, x_map, windowed) {
                AppNext::Menu => last_left = true, // swallow the click, stay in the menu
                other => return other,
            },
            Some(3) => return AppNext::Quit,
            _ => {}
        }
        t += 1;
    }
}

/// Single-player skirmish setup: choose how many AI opponents to face.
fn run_skirmish_setup(window: &mut Window, screen: &mut Canvas, canvas: &mut Canvas, x_map: &mut Vec<u32>, windowed: bool) -> AppNext {
    let mut t = 0u64;
    let mut last_left = window.get_mouse_down(MouseButton::Left);
    let mut last_mouse = v2(canvas.w as f32 / 2.0, canvas.h as f32 / 2.0);
    let accent = rgb(120, 230, 140);
    loop {
        if !window.is_open() {
            return AppNext::Quit;
        }
        sync_canvas(window, screen, canvas, x_map);
        if let Some((rx, ry)) = window.get_mouse_pos(MouseMode::Discard) {
            last_mouse = v2(rx * canvas.w as f32 / screen.w.max(1) as f32, ry * canvas.h as f32 / screen.h.max(1) as f32);
        }
        let mouse = last_mouse;
        let left = window.get_mouse_down(MouseButton::Left);
        let click = left && !last_left;
        last_left = left;

        draw_menu_bg(canvas, t, "SKIRMISH", "HOW MANY RIVALS? (FREE-FOR-ALL)", accent);
        let cx = canvas.w / 2;
        let labels = ["1 AI  (DUEL)", "2 AI  (3-WAY)", "3 AI  (4-WAY)"];
        let (bw, bh, gap) = (380, 56, 18);
        let y0 = canvas.h / 2 - 80;
        let mut chosen: Option<usize> = None;
        for (i, lab) in labels.iter().enumerate() {
            let b = Btn::new(cx, y0 + i as i32 * (bh + gap), bw, bh, lab, accent);
            let hover = b.hit(mouse);
            b.draw(canvas, hover);
            if click && hover {
                chosen = Some(i);
            }
        }
        let back = Btn::new(cx, y0 + 3 * (bh + gap) + 8, 220, 46, "BACK  [ESC]", rgb(180, 190, 196));
        let bhover = back.hit(mouse);
        back.draw(canvas, bhover);
        if click && bhover {
            return AppNext::Menu;
        }
        for k in window.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Key1 | Key::NumPad1 => chosen = Some(0),
                Key::Key2 | Key::NumPad2 => chosen = Some(1),
                Key::Key3 | Key::NumPad3 => chosen = Some(2),
                Key::Escape => return AppNext::Menu,
                _ => {}
            }
        }
        present_frame(window, screen, canvas, x_map);
        if !windowed {
            fullscreen_tick(t);
        }
        if let Some(i) = chosen {
            return AppNext::Single(i + 2); // factions = opponents + 1 human
        }
        t += 1;
    }
}

/// The join screen: live LAN host discovery (clickable) plus a typed-address
/// field. Returns Join(addr), Menu (back), or Quit.
fn run_join_browse(window: &mut Window, screen: &mut Canvas, canvas: &mut Canvas, x_map: &mut Vec<u32>, windowed: bool) -> AppNext {
    let disco = net::Discover::start().ok();
    let mut hosts: Vec<String> = Vec::new();
    let mut addr = String::new();
    let mut t = 0u64;
    let mut last_left = window.get_mouse_down(MouseButton::Left);
    let mut last_mouse = v2(canvas.w as f32 / 2.0, canvas.h as f32 / 2.0);
    let accent = rgb(255, 200, 90);
    loop {
        if !window.is_open() {
            return AppNext::Quit;
        }
        sync_canvas(window, screen, canvas, x_map);
        if let Some(d) = &disco {
            if t % 20 == 0 {
                d.ping();
            }
            if let Some(a) = d.poll() {
                let s = a.to_string();
                if !hosts.contains(&s) {
                    hosts.push(s);
                }
            }
        }
        if let Some((rx, ry)) = window.get_mouse_pos(MouseMode::Discard) {
            last_mouse = v2(rx * canvas.w as f32 / screen.w.max(1) as f32, ry * canvas.h as f32 / screen.h.max(1) as f32);
        }
        let mouse = last_mouse;
        let left = window.get_mouse_down(MouseButton::Left);
        let click = left && !last_left;
        last_left = left;

        let mut connect: Option<String> = None;
        for k in window.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Escape => return AppNext::Menu,
                Key::Backspace => {
                    addr.pop();
                }
                Key::Enter | Key::NumPadEnter => {
                    if !addr.is_empty() {
                        connect = Some(normalize_addr(&addr));
                    }
                }
                _ => {
                    if let Some(ch) = key_char(k) {
                        if addr.len() < 23 {
                            addr.push(ch);
                        }
                    }
                }
            }
        }

        let sub = if hosts.is_empty() { "SCANNING THE LAN FOR HOSTS" } else { "SELECT A HOST" };
        draw_menu_bg(canvas, t, "JOIN GAME", sub, accent);
        let cx = canvas.w / 2;
        let (bw, bh, gap) = (480, 48, 12);
        let mut y = canvas.h / 2 - 130;
        for h in hosts.iter().take(5) {
            let b = Btn::new(cx, y, bw, bh, h, rgb(120, 230, 150));
            let hover = b.hit(mouse);
            b.draw(canvas, hover);
            if click && hover {
                connect = Some(h.clone());
            }
            y += bh + gap;
        }
        if hosts.is_empty() {
            let dots = ".".repeat((t as usize / 16) % 4 + 1);
            canvas.text_center(cx, y + 8, &format!("LOOKING{}", dots), rgb(150, 165, 170), 2);
        }
        // Manual address entry.
        let fy = canvas.h - 196;
        canvas.text_center(cx, fy - 26, "OR TYPE AN ADDRESS  (IP:PORT)  THEN ENTER", rgb(150, 165, 170), 2);
        let bx = cx - bw / 2;
        canvas.fill_rect_a(bx, fy, bw, 48, rgb(10, 14, 12), 0.9);
        canvas.rect(bx, fy, bw, 48, accent);
        let caret = if (t / 28) % 2 == 0 { "_" } else { " " };
        let shown = format!("{}{}", addr, caret);
        canvas.text(bx + 14, fy + 14, &shown, rgb(230, 240, 240), 3);
        let back = Btn::new(cx, canvas.h - 92, 220, 44, "BACK  [ESC]", rgb(180, 190, 196));
        let bhover = back.hit(mouse);
        back.draw(canvas, bhover);
        if click && bhover {
            return AppNext::Menu;
        }

        present_frame(window, screen, canvas, x_map);
        if !windowed {
            fullscreen_tick(t);
        }
        if let Some(a) = connect {
            return AppNext::Join(a);
        }
        t += 1;
    }
}

/// Host lobby: pick a player count, beacon on the LAN, accept joiners, and start
/// when ready (empty slots become AI). Returns the live lockstep link, or None.
fn run_host_lobby(window: &mut Window, screen: &mut Canvas, canvas: &mut Canvas, x_map: &mut Vec<u32>, windowed: bool, port: u16) -> Option<net::Lockstep> {
    let mut host = match net::Host::bind(port, seed(), 4) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("could not host on port {port}: {e}");
            return None;
        }
    };
    let ip = net::local_ipv4().map(|i| format!("{}:{}", i, port)).unwrap_or_else(|| format!("port {}", port));
    let accent = rgb(90, 200, 255);
    let mut t = 0u64;
    let mut last_left = window.get_mouse_down(MouseButton::Left);
    let mut last_mouse = v2(canvas.w as f32 / 2.0, canvas.h as f32 / 2.0);
    loop {
        if !window.is_open() || window.is_key_pressed(Key::Escape, KeyRepeat::No) {
            return None;
        }
        sync_canvas(window, screen, canvas, x_map);
        host.poll();
        if let Some((rx, ry)) = window.get_mouse_pos(MouseMode::Discard) {
            last_mouse = v2(rx * canvas.w as f32 / screen.w.max(1) as f32, ry * canvas.h as f32 / screen.h.max(1) as f32);
        }
        let mouse = last_mouse;
        let left = window.get_mouse_down(MouseButton::Left);
        let click = left && !last_left;
        last_left = left;

        let players = host.players();
        draw_menu_bg(canvas, t, "HOSTING", &ip, accent);
        let cx = canvas.w / 2;
        // Slot list: 0 = you, 1..3 = joined / waiting / (AI on start).
        let mut y = canvas.h / 2 - 150;
        for f in 0..4 {
            let (label, col) = if f == 0 {
                (format!("SLOT {}:  YOU  ({})", f + 1, faction_name(Team::from_idx(f))), rgb(120, 230, 150))
            } else if f < players {
                (format!("SLOT {}:  PLAYER  ({})", f + 1, faction_name(Team::from_idx(f))), rgb(150, 200, 255))
            } else {
                (format!("SLOT {}:  OPEN -> AI", f + 1), rgb(120, 130, 140))
            };
            canvas.text_center(cx, y, &label, col, 3);
            y += 40;
        }
        canvas.text_center(cx, y + 8, &format!("{} PLAYER{} CONNECTED  -  LAN DISCOVERY ON", players, if players == 1 { "" } else { "S" }), rgb(150, 165, 170), 2);

        let start = Btn::new(cx, canvas.h - 150, 340, 56, "START MATCH", rgb(120, 230, 150));
        let shover = start.hit(mouse);
        start.draw(canvas, shover);
        canvas.text_center(cx, canvas.h - 78, "EMPTY SLOTS BECOME AI  -  ESC TO CANCEL", rgb(150, 165, 170), 2);
        let begin = (click && shover) || window.is_key_pressed(Key::Enter, KeyRepeat::No);

        present_frame(window, screen, canvas, x_map);
        if !windowed {
            fullscreen_tick(t);
        }
        if begin {
            return Some(host.start());
        }
        t += 1;
    }
}

/// Connect to a host (explicit address or LAN auto-discovery), then wait in the
/// host's lobby for the match to start. Returns the live lockstep link, or None.
fn run_join_connect(window: &mut Window, screen: &mut Canvas, canvas: &mut Canvas, x_map: &mut Vec<u32>, windowed: bool, addr: &str) -> Option<net::Lockstep> {
    let target = if addr == "auto" {
        let disco = net::Discover::start().ok();
        let mut found: Option<String> = None;
        let mut t = 0u64;
        loop {
            if !window.is_open() || window.is_key_pressed(Key::Escape, KeyRepeat::No) {
                return None;
            }
            sync_canvas(window, screen, canvas, x_map);
            if let Some(d) = &disco {
                if t % 24 == 0 {
                    d.ping();
                }
                if let Some(a) = d.poll() {
                    found = Some(a.to_string());
                }
            }
            if let Some(f) = &found {
                draw_lobby(canvas, t, "CHALLENGER", "HOST FOUND", f, "connecting...", rgb(120, 240, 150));
                present_frame(window, screen, canvas, x_map);
                break;
            }
            let dots = ".".repeat((t as usize / 16) % 4 + 1);
            draw_lobby(canvas, t, "CHALLENGER", &format!("SCANNING LAN{}", dots), "", "searching for a host  -  Esc to cancel", rgb(255, 190, 90));
            present_frame(window, screen, canvas, x_map);
            if !windowed {
                fullscreen_tick(t);
            }
            t += 1;
        }
        found?
    } else {
        addr.to_string()
    };
    sync_canvas(window, screen, canvas, x_map);
    draw_lobby(canvas, 1, "CHALLENGER", "CONNECTING", &target, "", rgb(120, 240, 150));
    present_frame(window, screen, canvas, x_map);
    let mut joiner = match net::Joiner::connect(&target) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("could not join {target}: {e}");
            return None;
        }
    };
    // In the lobby: wait for the host to start the match.
    let me = Team::from_idx(joiner.my_faction);
    let mut t = 0u64;
    loop {
        if !window.is_open() || window.is_key_pressed(Key::Escape, KeyRepeat::No) {
            return None;
        }
        sync_canvas(window, screen, canvas, x_map);
        if let Some(ls) = joiner.poll_start() {
            return Some(ls);
        }
        if !joiner.alive() {
            draw_lobby(canvas, t, "DISCONNECTED", "HOST LEFT", "", "Esc to go back", rgb(235, 120, 110));
            present_frame(window, screen, canvas, x_map);
            if window.is_key_pressed(Key::Escape, KeyRepeat::No) {
                return None;
            }
            t += 1;
            continue;
        }
        draw_lobby(
            canvas,
            t,
            "IN LOBBY",
            &format!("YOU ARE {}", faction_name(me)),
            "WAITING FOR HOST TO START",
            "Esc to leave",
            fill_of(me),
        );
        present_frame(window, screen, canvas, x_map);
        if !windowed {
            fullscreen_tick(t);
        }
        t += 1;
    }
}

/// Centre the camera on the local player's base and announce the match.
fn intro_world(w: &mut World, canvas: &Canvas) {
    if let Some(b) = w.ents.iter().find(|e| e.kind == Kind::Base && e.team == w.my_team) {
        let p = b.pos;
        w.cam = v2(p.x - canvas.w as f32 / 2.0, p.y - canvas.h as f32 / 2.0);
        w.clamp_cam(canvas.w as f32, canvas.h as f32);
    }
    w.messages.clear();
    let foes = w.factions - 1;
    w.msg(&format!(
        "YOU ARE {} - {} RIVAL{} - LAST ONE STANDING WINS",
        faction_name(w.my_team),
        foes,
        if foes == 1 { "" } else { "S" },
    ));
}

/// A fresh single-player free-for-all: faction 0 is the human, the rest are AI.
fn make_world_sp(factions: usize, canvas: &Canvas) -> World {
    let mut is_ai = [false; world::MAX_FACTIONS];
    for slot in is_ai.iter_mut().take(factions).skip(1) {
        *slot = true;
    }
    let mut w = World::new_match(seed(), factions, is_ai, Team::Player, false);
    intro_world(&mut w, canvas);
    w
}

/// The multiplayer world built from a lockstep session's shared config.
fn make_world_mp(ls: &net::Lockstep, canvas: &Canvas) -> World {
    let mut is_ai = [false; world::MAX_FACTIONS];
    for fi in 0..ls.factions {
        is_ai[fi] = ls.is_ai[fi];
    }
    let mut w = World::new_match(ls.seed, ls.factions, is_ai, Team::from_idx(ls.my_faction), true);
    intro_world(&mut w, canvas);
    w
}

/// Apply one lockstep step's commands: faction `f`'s list to `Team::from_idx(f)`.
/// (AI factions have empty lists — their orders come from the local AI.)
fn apply_step(w: &mut World, per_faction: &[Vec<world::Cmd>]) {
    for (f, cmds) in per_faction.iter().enumerate() {
        let team = Team::from_idx(f);
        for c in cmds {
            w.apply_cmd(team, c);
        }
    }
}

/// Pause-menu buttons (also the keyboard shortcuts they mirror).
fn pause_buttons(c: &Canvas) -> Vec<(Btn, MenuAct)> {
    let cx = c.w / 2;
    let (bw, bh, gap) = (320, 50, 14);
    let y0 = c.h / 2 - 64;
    vec![
        (Btn::new(cx, y0, bw, bh, "RESUME  [ESC]", rgb(120, 230, 150)), MenuAct::Resume),
        (Btn::new(cx, y0 + (bh + gap), bw, bh, "MAIN MENU  [M]", rgb(90, 200, 255)), MenuAct::ToMenu),
        (Btn::new(cx, y0 + 2 * (bh + gap), bw, bh, "SURRENDER  [G]", rgb(240, 200, 110)), MenuAct::Surrender),
        (Btn::new(cx, y0 + 3 * (bh + gap), bw, bh, "QUIT  [Q]", rgb(240, 130, 120)), MenuAct::Quit),
    ]
}

/// Game-over buttons. Play Again is single-player only.
fn gameover_buttons(c: &Canvas, versus: bool) -> Vec<(Btn, MenuAct)> {
    let cx = c.w / 2;
    let (bw, bh, gap) = (320, 50, 14);
    let mut y = c.h / 2 + 56;
    let mut v = Vec::new();
    if !versus {
        v.push((Btn::new(cx, y, bw, bh, "PLAY AGAIN  [R]", rgb(120, 230, 150)), MenuAct::Restart));
        y += bh + gap;
    }
    v.push((Btn::new(cx, y, bw, bh, "MAIN MENU  [M]", rgb(90, 200, 255)), MenuAct::ToMenu));
    y += bh + gap;
    v.push((Btn::new(cx, y, bw, bh, "QUIT  [Q]", rgb(240, 130, 120)), MenuAct::Quit));
    v
}

/// Ask the window manager to make our window fullscreen.
///
/// On Linux minifb can't do this itself, so we shell out: `wmctrl` for most EWMH
/// window managers, `xdotool`'s FULLSCREEN state for wlroots/Smithay XWayland
/// (where wmctrl's request is silently ignored). We try both by window id and
/// fall back to matching by name — whichever tool is missing is harmless.
///
/// Other platforms don't have these tools; there we open a borderless window
/// already sized to the monitor (see `main`), so this is a no-op.
#[cfg(target_os = "linux")]
fn request_fullscreen() {
    let _ = std::process::Command::new("sh")
        .arg("-c")
        .arg(
            "wid=$(xdotool search --name '^nanorts$' 2>/dev/null | head -n1); \
             if [ -n \"$wid\" ]; then \
               xdotool windowstate --add FULLSCREEN \"$wid\" 2>/dev/null; \
               wmctrl -i -r \"$wid\" -b add,fullscreen 2>/dev/null; \
             else \
               wmctrl -r nanorts -b add,fullscreen 2>/dev/null; \
             fi",
        )
        .spawn();
}

#[cfg(not(target_os = "linux"))]
fn request_fullscreen() {}

/// Fire the fullscreen request a handful of times over the first couple of
/// seconds — the window must be mapped and named before any tool can find it,
/// and the request is idempotent, so retrying costs nothing.
fn fullscreen_tick(frame: u64) {
    if matches!(frame, 10 | 45 | 90 | 160) {
        request_fullscreen();
    }
}

fn main() {
    // Headless determinism / sanity harness: `nanorts --sim`. Runs the full
    // simulation with no window so the game logic can be exercised on a box
    // with no display.
    if std::env::args().any(|a| a == "--sim") {
        run_headless();
        return;
    }

    // Headless lockstep determinism test over loopback:
    //   nanorts --mptest-host 7777   (run first, waits for a peer)
    //   nanorts --mptest-join 127.0.0.1:7777
    let args: Vec<String> = std::env::args().collect();
    if let Some(p) = args.iter().position(|a| a == "--mptest-host") {
        let port: u16 = args.get(p + 1).and_then(|s| s.parse().ok()).unwrap_or(7777);
        mptest(net::Lockstep::host(port, seed()), true);
        return;
    }
    if let Some(p) = args.iter().position(|a| a == "--mptest-join") {
        let addr = args.get(p + 1).cloned().unwrap_or_else(|| "127.0.0.1:7777".into());
        mptest(net::Lockstep::join(&addr), false);
        return;
    }
    // In-process 4-human-peer determinism test (host + 3 joiners over loopback,
    // exercising the relay): `nanorts --mptest4`.
    if args.iter().any(|a| a == "--mptest4") {
        mptest4();
        return;
    }

    // `--windowed` / `-w`: a normal resizable window instead of fullscreen
    // (smaller buffer = easy 60 fps; also handy for testing).
    let windowed = std::env::args().any(|a| a == "--windowed" || a == "-w");

    // Multiplayer mode: --host [port] or --join <ip:port>.
    let mp_host_port: Option<u16> = args
        .iter()
        .position(|a| a == "--host")
        .map(|p| args.get(p + 1).and_then(|s| s.parse().ok()).unwrap_or(7777));
    // `--join <addr>` dials an explicit address; bare `--join` (or `--join auto`)
    // hunts for a host on the LAN via UDP discovery.
    let mp_join_addr: Option<String> = args.iter().position(|a| a == "--join").map(|p| {
        match args.get(p + 1) {
            Some(a) if !a.starts_with('-') => a.clone(),
            _ => "auto".into(),
        }
    });

    // On Linux, prefer the X11 backend when available: it lets the WM full-screen
    // us and gives reliable behaviour (native Wayland clients can't self-position).
    #[cfg(target_os = "linux")]
    if std::env::var_os("DISPLAY").is_some() {
        std::env::remove_var("WAYLAND_DISPLAY");
    }
    let (fsw, fsh, fsx, fsy) = detect_screen();
    let fullscreen = !windowed;
    // Fullscreen means a borderless window filling the monitor on Linux/Windows.
    // macOS has no scriptable fullscreen and a borderless window fights the menu
    // bar, so there we open a large *titled, resizable* window instead — use the
    // green traffic-light button for the OS's own fullscreen (the game rescales
    // live, so it adapts the moment you toggle it).
    let mac = cfg!(target_os = "macos");
    let borderless = fullscreen && !mac;
    let (sw, sh) = if !fullscreen {
        (1600, 900)
    } else if mac {
        // A big window that still leaves room for the menu bar / title bar.
        ((fsw as f32 * 0.9) as usize, (fsh as f32 * 0.85) as usize)
    } else {
        (fsw, fsh)
    };
    // `canvas` is the design buffer we draw the game into; `screen` is the real
    // window-sized buffer we upscale into and present. The window is resizable
    // so the WM (or the user) can full-screen it; we track its size and rescale
    // each frame.
    let mut screen = Canvas::new(sw as i32, sh as i32);
    let mut canvas = Canvas::new(design_width(sw as i32, sh as i32), DESIGN_H);
    let mut x_map = build_x_map(screen.w, canvas.w);
    let mut window = Window::new(
        "nanorts",
        sw,
        sh,
        WindowOptions {
            borderless,
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .expect("could not open a window");
    window.set_target_fps(60);
    // A borderless fullscreen window with no WM to place it (Windows et al.) needs
    // pinning to the monitor's corner so it covers the screen. Linux leaves it to
    // the WM; macOS uses a titled window, so neither positions here.
    if borderless && !cfg!(target_os = "linux") {
        window.set_position(fsx as isize, fsy as isize);
    }

    // Procedural audio — silent if no device is available.
    let audio = audio::Audio::new();

    // The app is a state machine: main menu -> (mp setup) -> a match -> back to
    // the menu. CLI flags jump straight into a mode for back-compat.
    let mut next = if let Some(port) = mp_host_port {
        AppNext::Host(port)
    } else if let Some(addr) = mp_join_addr {
        AppNext::Join(addr)
    } else {
        AppNext::Menu
    };

    'app: loop {
        if !window.is_open() {
            break;
        }
        // Resolve the next state into a match to play (or loop back to the menu).
        let (mut net, mut world): (Option<net::Lockstep>, World) = match next {
            AppNext::Quit => break,
            AppNext::Menu => {
                next = run_main_menu(&mut window, &mut screen, &mut canvas, &mut x_map, windowed);
                continue 'app;
            }
            AppNext::Single(factions) => (None, make_world_sp(factions, &canvas)),
            AppNext::Host(port) => match run_host_lobby(&mut window, &mut screen, &mut canvas, &mut x_map, windowed, port) {
                Some(ls) => {
                    let w = make_world_mp(&ls, &canvas);
                    (Some(ls), w)
                }
                None => {
                    next = AppNext::Menu;
                    continue 'app;
                }
            },
            AppNext::Join(addr) => match run_join_connect(&mut window, &mut screen, &mut canvas, &mut x_map, windowed, &addr) {
                Some(ls) => {
                    let w = make_world_mp(&ls, &canvas);
                    (Some(ls), w)
                }
                None => {
                    next = AppNext::Menu;
                    continue 'app;
                }
            },
        };

        let mut ui = Ui {
            drag_start: None,
            dragging: false,
            build_mode: None,
            attack_pending: false,
            show_help: false,
            menu: false,
        };
        let mut done: Option<GameOutcome> = None;
        // Seed from the live button state so a click still held from the menu
        // (e.g. the one that chose "Single Player") isn't read as a fresh
        // in-game press on frame 1.
        let mut prev_left = window.get_mouse_down(MouseButton::Left);
        let mut prev_right = window.get_mouse_down(MouseButton::Right);
        // Fixed-timestep accumulator: the sim always advances in 1/60s steps, but we
        // run as many steps per frame as real elapsed time calls for — so the game
        // runs at the correct speed even when rendering can't keep up at 60 fps.
        const STEP: f32 = 1.0 / 60.0;
        let mut accumulator = 0.0f32;
        let mut last_tick = Instant::now();

        // Persistent input state.
        let mut last_mouse = v2(sw as f32 / 2.0, sh as f32 / 2.0);
        let mut mid_pan: Option<V2> = None;
        let mut groups: Vec<Vec<u32>> = vec![Vec::new(); 10];
        let mut last_click_frame: u64 = 0;
        let mut last_click_pos = v2(-999.0, -999.0);
        let mut last_recall: Option<usize> = None;
        let mut last_recall_frame: u64 = 0;
        let mut frame_no: u64 = 0;

        while window.is_open() && done.is_none() {
            frame_no += 1;
            // Real elapsed time this frame, fed to the sim accumulator and used for
            // framerate-independent camera panning.
            let now = Instant::now();
            let frame_dt = (now - last_tick).as_secs_f32().min(0.25);
            last_tick = now;
            accumulator += frame_dt;

            // Track the live window size (the WM may resize us, e.g. on fullscreen)
            // and rebuild the screen buffer, design buffer, and upscale map to match.
            let (cw, ch) = window.get_size();
            if cw > 0 && ch > 0 && (cw as i32 != screen.w || ch as i32 != screen.h) {
                screen = Canvas::new(cw as i32, ch as i32);
                canvas = Canvas::new(design_width(cw as i32, ch as i32), DESIGN_H);
                x_map = build_x_map(screen.w, canvas.w);
            }
            // Tell the sim how big the local view is, so it can gate screen-wide
            // explosion flash/shake to blasts on or near the screen.
            world.view_w = canvas.w as f32;
            world.view_h = canvas.h as f32;

            // ---------------- input snapshot ----------------
            // Discard mode gives None when the cursor leaves the window — used for
            // selection/commands so a click never lands on a stale position.
            let raw = window.get_mouse_pos(MouseMode::Discard);
            if let Some((rx, ry)) = raw {
                let dx = rx * canvas.w as f32 / screen.w.max(1) as f32;
                let dy = ry * canvas.h as f32 / screen.h.max(1) as f32;
                last_mouse = v2(dx, dy);
            }
            let mouse = last_mouse;
            let (mx, my) = (mouse.x, mouse.y);
            // Clamp mode always reports a position pinned to the window edge — even
            // while the cursor is pushed past it — so edge-scrolling works like a
            // grabbed mouse (the window can't actually confine the cursor itself).
            let focused = window.is_active();
            let pan_mouse = window
                .get_mouse_pos(MouseMode::Clamp)
                .map(|(rx, ry)| v2(rx * canvas.w as f32 / screen.w.max(1) as f32, ry * canvas.h as f32 / screen.h.max(1) as f32))
                .unwrap_or(mouse);
            let left = window.get_mouse_down(MouseButton::Left);
            let right = window.get_mouse_down(MouseButton::Right);
            let middle = window.get_mouse_down(MouseButton::Middle);
            let shift =
                window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift);
            let ctrl =
                window.is_key_down(Key::LeftCtrl) || window.is_key_down(Key::RightCtrl);
            let keys = window.get_keys_pressed(KeyRepeat::No);
            let wm = v2(mx + world.cam.x, my + world.cam.y);

            // Player actions accumulate here, then are either applied immediately
            // (single-player) or queued for the lockstep network (multiplayer).
            let mut frame_cmds: Vec<world::Cmd> = Vec::new();

            // ---------------- hotkeys ----------------
            for k in keys {
                // Pause menu swallows all other input.
                if ui.menu {
                    match k {
                        Key::Escape => ui.menu = false,             // resume
                        Key::G => {
                            world.over = -1; // surrender -> defeat
                            ui.menu = false;
                        }
                        Key::M => done = Some(GameOutcome::ToMenu),
                        Key::Q => done = Some(GameOutcome::Quit),
                        _ => {}
                    }
                    continue;
                }
                // Game-over screen: restart, return to menu, or quit.
                if world.over != 0 {
                    match k {
                        // Restart only makes sense single-player; in versus both
                        // sims must stay identical, so there's no local reset.
                        // Reuse the current match's player count so "play again"
                        // keeps the 3-/4-way setup you chose.
                        Key::R if net.is_none() => {
                            world = make_world_sp(world.factions, &canvas);
                            ui.build_mode = None;
                            ui.attack_pending = false;
                            ui.dragging = false;
                        }
                        Key::M => done = Some(GameOutcome::ToMenu),
                        Key::Q => done = Some(GameOutcome::Quit),
                        _ => {}
                    }
                    continue;
                }
                // Control groups: Ctrl+N assigns, N recalls, double-tap N centers.
                if let Some(n) = digit_of(k) {
                    if ctrl {
                        groups[n] = world.selected_ids();
                        if !groups[n].is_empty() {
                            world.msg(&format!("CONTROL GROUP {} SET", n));
                        }
                    } else {
                        world.select_ids(&groups[n], false);
                        if last_recall == Some(n)
                            && frame_no.wrapping_sub(last_recall_frame) < 20
                        {
                            if let Some(c) = world.centroid_of_ids(&groups[n]) {
                                world.cam = v2(c.x - canvas.w as f32 / 2.0, c.y - canvas.h as f32 / 2.0);
                                world.clamp_cam(canvas.w as f32, canvas.h as f32);
                            }
                        }
                        last_recall = Some(n);
                        last_recall_frame = frame_no;
                    }
                    continue;
                }
                // Ctrl+A grabs the whole army (every combat unit, no workers).
                if ctrl && k == Key::A {
                    world.select_all_army(shift);
                    if !world.selected_ids().is_empty() {
                        if let Some(a) = &audio {
                            a.play(audio::Sfx::Select, 1.0, 0.0);
                        }
                    }
                    continue;
                }
                match k {
                    // Train from selected production buildings.
                    Key::W => queue_train(&world, &mut frame_cmds, Kind::Base, Kind::Worker),
                    Key::E => queue_train(&world, &mut frame_cmds, Kind::Barracks, Kind::Soldier),
                    Key::Y => queue_train(&world, &mut frame_cmds, Kind::Barracks, Kind::Pyro),
                    Key::G => queue_train(&world, &mut frame_cmds, Kind::Barracks, Kind::Sapper),
                    Key::T => queue_train(&world, &mut frame_cmds, Kind::Factory, Kind::Tank),
                    Key::R => queue_train(&world, &mut frame_cmds, Kind::Factory, Kind::Raider),
                    Key::V => queue_train(&world, &mut frame_cmds, Kind::Factory, Kind::Mortar),
                    // Enter build-placement mode (needs a worker selected).
                    Key::B | Key::F | Key::D | Key::C => {
                        let kind = match k {
                            Key::B => Kind::Barracks,
                            Key::F => Kind::Factory,
                            Key::D => Kind::Depot,
                            _ => Kind::Base,
                        };
                        if has_player_worker(&world) {
                            ui.build_mode = Some(kind);
                            ui.attack_pending = false;
                            world.msg(&format!("LEFT-CLICK TO PLACE {} (SHIFT-CLICK CHAINS)", kind_name(kind)));
                        } else {
                            world.msg("SELECT A WORKER FIRST");
                        }
                    }
                    Key::A => {
                        let has_mil = world.ents.iter().any(|e| {
                            e.selected
                                && e.team == Team::Player
                                && (world::is_army(e.kind) || e.kind == Kind::Worker)
                        });
                        if has_mil {
                            ui.attack_pending = true;
                            ui.build_mode = None;
                        }
                    }
                    Key::S => frame_cmds.push(world::Cmd::Stop { ids: world.selected_ids() }),
                    Key::H => ui.show_help = !ui.show_help,
                    Key::Escape => {
                        // Cancel an active targeting mode first; otherwise open the
                        // pause menu (deselect with a click on empty ground).
                        if ui.build_mode.is_some() || ui.attack_pending {
                            ui.build_mode = None;
                            ui.attack_pending = false;
                        } else {
                            ui.menu = true;
                            ui.dragging = false;
                        }
                    }
                    _ => {}
                }
            }

            // ---------------- mouse buttons ----------------
            let left_pressed = left && !prev_left;
            let left_released = !left && prev_left;
            let right_pressed = right && !prev_right;
            // Snapshot now: a Restart click below clears `world.over` mid-frame,
            // and the gameplay handlers must still treat this frame as an overlay
            // click (so it doesn't also start a stray drag-select in the new game).
            let in_overlay = ui.menu || world.over != 0;

            // Clickable pause / game-over buttons (mirror the keyboard shortcuts).
            if in_overlay && left_pressed {
                let buttons = if ui.menu {
                    pause_buttons(&canvas)
                } else {
                    gameover_buttons(&canvas, world.versus)
                };
                for (b, act) in buttons {
                    if b.hit(mouse) {
                        match act {
                            MenuAct::Resume => ui.menu = false,
                            MenuAct::Surrender => {
                                world.over = -1;
                                ui.menu = false;
                            }
                            MenuAct::Restart if net.is_none() => {
                                world = make_world_sp(world.factions, &canvas);
                                ui.build_mode = None;
                                ui.attack_pending = false;
                                ui.dragging = false;
                            }
                            MenuAct::Restart => {}
                            MenuAct::ToMenu => done = Some(GameOutcome::ToMenu),
                            MenuAct::Quit => done = Some(GameOutcome::Quit),
                        }
                    }
                }
            }

            if !in_overlay && left && in_minimap(mouse, canvas.w, canvas.h) {
                let wp = mm_to_world(mouse, canvas.w, canvas.h, world.world_w, world.world_h);
                if left_pressed && ui.attack_pending {
                    // A + minimap click = attack-move across the map.
                    let tgt = world.snap_to_enemy(wp, 130.0).unwrap_or(wp);
                    frame_cmds.push(world::Cmd::Order {
                        ids: world.selected_ids(),
                        x: tgt.x,
                        y: tgt.y,
                        attack_move: true,
                        queue: shift,
                    });
                    ui.attack_pending = false;
                } else {
                    // Otherwise click / drag re-centers the camera.
                    world.cam = v2(wp.x - canvas.w as f32 / 2.0, wp.y - canvas.h as f32 / 2.0);
                    world.clamp_cam(canvas.w as f32, canvas.h as f32);
                }
            } else {
                if !in_overlay && left_pressed && !in_hud(mouse) && !in_minimap(mouse, canvas.w, canvas.h) {
                    if ui.attack_pending {
                        frame_cmds.push(world::Cmd::Order {
                            ids: world.selected_ids(),
                            x: wm.x,
                            y: wm.y,
                            attack_move: true,
                            queue: shift,
                        });
                        ui.attack_pending = false;
                    } else if let Some(k) = ui.build_mode {
                        if !world.can_build(k, wm) {
                            world.msg("CANT BUILD THERE");
                        } else if let Some(wid) = first_selected_worker(&world) {
                            // Hold Ctrl to chain: queue this build and stay in
                            // placement mode for the next one.
                            frame_cmds.push(world::Cmd::Build { worker: wid, kind: k, x: wm.x, y: wm.y, chain: shift });
                            if shift {
                                world.msg(&format!("QUEUED {} - SHIFT-CLICK FOR MORE", kind_name(k)));
                            } else {
                                ui.build_mode = None;
                            }
                        } else {
                            world.msg("SELECT A WORKER FIRST");
                        }
                    } else {
                        ui.drag_start = Some(wm);
                        ui.dragging = true;
                    }
                }
                if !in_overlay && left_released && ui.dragging {
                    if let Some(ds) = ui.drag_start {
                        if ds.dist(wm) < 6.0 {
                            // A click, not a drag. Detect double-click (a forgiving
                            // ~0.4s window and a loose radius so it fires reliably).
                            let dbl = frame_no.wrapping_sub(last_click_frame) < 26
                                && last_click_pos.dist(wm) < 18.0;
                            if dbl {
                                let vmin = world.cam;
                                let vmax = v2(world.cam.x + canvas.w as f32, world.cam.y + canvas.h as f32);
                                world.select_type_in_view(wm, vmin, vmax, shift);
                            } else {
                                world.select_single(wm, shift);
                            }
                            last_click_frame = frame_no;
                            last_click_pos = wm;
                        } else {
                            world.select_box(ds, wm, shift);
                        }
                        if !world.selected_ids().is_empty() {
                            if let Some(a) = &audio {
                                a.play(audio::Sfx::Select, 1.0, 0.0); // UI: always centered/full
                            }
                        }
                    }
                    ui.dragging = false;
                    ui.drag_start = None;
                }
            }

            if !in_overlay && right_pressed {
                if ui.build_mode.is_some() || ui.attack_pending {
                    // Right-click cancels an active placement / attack-target mode —
                    // it must NOT also issue a move, or it would yank the selected
                    // worker off the build chain you were queueing.
                    ui.build_mode = None;
                    ui.attack_pending = false;
                } else {
                    let tgt = if in_minimap(mouse, canvas.w, canvas.h) {
                        let wp = mm_to_world(mouse, canvas.w, canvas.h, world.world_w, world.world_h);
                        Some(world.snap_to_enemy(wp, 130.0).unwrap_or(wp))
                    } else if !in_hud(mouse) {
                        Some(wm)
                    } else {
                        None
                    };
                    if let Some(t) = tgt {
                        frame_cmds.push(world::Cmd::Order {
                            ids: world.selected_ids(),
                            x: t.x,
                            y: t.y,
                            attack_move: false,
                            queue: shift,
                        });
                    }
                }
            }

            // ---------------- camera pan ----------------
            // Middle-mouse grabs and drags the map. Use the clamped position so a
            // drag that runs off the window edge pauses cleanly instead of jumping.
            let mut panning = false;
            if middle && focused && !ui.menu {
                if let Some(prev) = mid_pan {
                    world.cam = world.cam.sub(pan_mouse.sub(prev));
                }
                mid_pan = Some(pan_mouse);
                panning = true;
            } else {
                mid_pan = None;
            }

            // Frozen while paused.
            let pan = if ui.menu { 0.0 } else { 760.0 * frame_dt };
            // Arrow keys always pan.
            if window.is_key_down(Key::Left) {
                world.cam.x -= pan;
            }
            if window.is_key_down(Key::Right) {
                world.cam.x += pan;
            }
            if window.is_key_down(Key::Up) {
                world.cam.y -= pan;
            }
            if window.is_key_down(Key::Down) {
                world.cam.y += pan;
            }
            // Edge-scroll: push the cursor to a screen edge to slide the view that
            // way. Driven by the clamped position so it keeps scrolling even as the
            // cursor leaves the window (the "grab" the OS won't give us). Only while
            // focused, not mid drag-select/map-grab, and not over the minimap.
            if focused && !panning && !ui.dragging && !in_minimap(pan_mouse, canvas.w, canvas.h) {
                let band = 18.0;
                let (ex, ey) = (pan_mouse.x, pan_mouse.y);
                if ex <= band {
                    world.cam.x -= pan;
                }
                if ex >= canvas.w as f32 - band {
                    world.cam.x += pan;
                }
                if ey <= band {
                    world.cam.y -= pan;
                }
                if ey >= canvas.h as f32 - band {
                    world.cam.y += pan;
                }
            }
            world.clamp_cam(canvas.w as f32, canvas.h as f32);

            prev_left = left;
            prev_right = right;

            // ---------------- route this frame's commands ----------------
            // Single-player applies them now; multiplayer hands them to the lockstep
            // network, which schedules them a few steps ahead and feeds them back to
            // both peers in the same step so the simulations stay identical.
            if let Some(ls) = net.as_mut() {
                for c in frame_cmds.drain(..) {
                    ls.queue(c);
                }
            } else {
                let mt = world.my_team;
                for c in frame_cmds.drain(..) {
                    world.apply_cmd(mt, &c);
                }
            }

            // ---------------- simulate & draw ----------------
            // Advance the sim by however many fixed steps real time demands, so the
            // game keeps real-time speed regardless of frame rate.
            if let Some(ls) = net.as_mut() {
                // Lockstep: a step runs only once we hold BOTH players' inputs for
                // it. If the peer is behind we simply wait (and don't drain time).
                if ls.alive() {
                    ls.service();
                    let mut steps = 0;
                    while accumulator >= STEP && steps < 8 {
                        if !ls.ready() {
                            break;
                        }
                        apply_step(&mut world, &ls.step_cmds());
                        world.update(STEP);
                        ls.advanced(world.checksum());
                        accumulator -= STEP;
                        steps += 1;
                        ls.service();
                    }
                }
                // Cap the backlog so a momentary stall doesn't cause a fast-forward.
                if accumulator > STEP * 10.0 {
                    accumulator = STEP * 10.0;
                }
            } else if ui.menu {
                accumulator = 0.0; // paused (single-player only)
            } else {
                let mut steps = 0;
                while accumulator >= STEP && steps < 8 {
                    world.update(STEP);
                    accumulator -= STEP;
                    steps += 1;
                }
            }
            // Hand this tick's sound events to the synth, faded by how far each
            // is from the current view (and panned across the stereo field).
            if let Some(a) = &audio {
                let vmin = world.cam;
                let vmax = v2(world.cam.x + canvas.w as f32, world.cam.y + canvas.h as f32);
                for (s, pos) in world.sounds.drain(..) {
                    let (gain, pan) = match pos {
                        None => (1.0, 0.0), // global UI cue
                        Some(p) => sound_at_view(vmin, vmax, p),
                    };
                    a.play(s, gain, pan);
                }
            } else {
                world.sounds.clear();
            }
            render(&mut canvas, &world, &ui, mouse);
            if let Some(ls) = &net {
                draw_sync_badge(&mut canvas, ls);
            }
            blit_scaled(&canvas, &mut screen, &x_map);
            // Only present when the screen buffer matches the window exactly (so we
            // never trigger minifb's scaler). On a size mismatch or a minimized /
            // undrawable window, just pump events and resync next frame — no panic.
            let (cw, ch) = window.get_size();
            if cw == screen.w as usize && ch == screen.h as usize && cw > 0 {
                let _ = window.update_with_buffer(&screen.buf, cw, ch);
            } else {
                window.update();
            }

            // Once mapped, ask the compositor to make us fullscreen (retried a few
            // times since the window must exist before any tool can find it).
            if !windowed {
                fullscreen_tick(frame_no);
            }
        }

        // The match ended (or the player chose to leave). Quit the app, or fall
        // back to the main menu for another game.
        match done {
            Some(GameOutcome::Quit) => break,
            _ => next = AppNext::Menu,
        }
    }
}

// ---- headless harness -----------------------------------------------------

struct Census {
    wk: u32,
    sol: u32,
    pyro: u32,
    tank: u32,
    raider: u32,
    base: u32,
    rax: u32,
    fac: u32,
    depot: u32,
}
fn census(w: &World, team: Team) -> Census {
    let mut c = Census { wk: 0, sol: 0, pyro: 0, tank: 0, raider: 0, base: 0, rax: 0, fac: 0, depot: 0 };
    for e in &w.ents {
        if e.team != team {
            continue;
        }
        match e.kind {
            Kind::Worker => c.wk += 1,
            Kind::Soldier => c.sol += 1,
            Kind::Pyro => c.pyro += 1,
            Kind::Tank => c.tank += 1,
            Kind::Raider => c.raider += 1,
            Kind::Base => c.base += 1,
            Kind::Barracks => c.rax += 1,
            Kind::Factory => c.fac += 1,
            Kind::Depot => c.depot += 1,
            _ => {}
        }
    }
    c
}

fn run_headless() {
    // Optional seed arg: `nanorts --sim 12345` to sample different personalities.
    let seed = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0xC0FFEE);
    let mut w = World::new(seed);
    let dt = 1.0 / 60.0;
    let max = 60 * 360; // 6 minutes of sim
    println!("nanorts headless sim (seed {seed}, player auto-defends)");
    let ai = &w.ai[1]; // the Enemy AI in the default 2-faction sim
    println!(
        "AI personality: {:?}  workers~{} aggression {:.0}% patience {:.1}s tank/pyro/raider {:.0}/{:.0}/{:.0}% expand@{} harass {}",
        ai.strategy,
        ai.worker_target,
        ai.aggression * 100.0,
        ai.patience,
        ai.tank_ratio * 100.0,
        ai.pyro_ratio * 100.0,
        ai.raider_ratio * 100.0,
        ai.expand_min,
        ai.harass,
    );
    println!("  t   | enemy: wk sol pyr tnk rdr | rax fac dep base | min sup | intent");
    for f in 0..max {
        // Give the player a defensive trickle so the AI faces real resistance —
        // this lets us watch it tech up, expand, push, and regroup.
        if f > 0 && f % (60 * 10) == 0 {
            if let Some(pb) = w.first_base(Team::Player) {
                let p = w.ents[pb].pos;
                for k in 0..3 {
                    let s = w.spawn(Kind::Soldier, Team::Player, p.add(v2(40.0 + k as f32 * 16.0, -50.0)));
                    let _ = s;
                }
            }
        }
        w.update(dt);
        if f % (60 * 15) == 0 || w.over != 0 {
            let e = census(&w, Team::Enemy);
            println!(
                "{:4}s | {:2} {:3} {:3} {:3} {:3} | {:2}  {:2}  {:2}  {:2}  | {:4} {:2}/{:<2} | {:?}",
                f / 60,
                e.wk, e.sol, e.pyro, e.tank, e.raider,
                e.rax, e.fac, e.depot, e.base,
                w.team_min(Team::Enemy),
                w.supply_used(Team::Enemy), w.supply_cap(Team::Enemy),
                w.ai[1].intent,
            );
        }
        if w.over != 0 {
            let r = match w.over {
                1 => "PLAYER WINS",
                -1 => "ENEMY WINS",
                _ => "?",
            };
            println!("=> game over at {}s: {}  (kills {})", f / 60, r, w.kills);
            return;
        }
    }
    let p = census(&w, Team::Player);
    let e = census(&w, Team::Enemy);
    println!(
        "=> no winner after {}s. player base {} / enemy base {}",
        max / 60,
        p.base,
        e.base
    );
}

/// Headless lockstep determinism test: two peers run the identical sim driven
/// only by exchanged commands, comparing periodic checksums. If determinism
/// holds (it must, for netcode to work) both print IN SYNC.
fn mptest(conn: std::io::Result<net::Lockstep>, host: bool) {
    let tag = if host { "HOST" } else { "JOIN" };
    let mut ls = match conn {
        Ok(ls) => ls,
        Err(e) => {
            eprintln!("{tag}: connection failed: {e}");
            return;
        }
    };
    println!("{tag}: connected, shared seed {:016x}", ls.seed);
    let mut w = World::new(ls.seed);
    w.versus = true;
    w.my_team = if host { Team::Player } else { Team::Enemy };
    let my_team = w.my_team;
    let step = 1.0 / 60.0;
    let max = 1800u32; // 30s of simulation
    let mut queued_train = false;
    let mut queued_build = false;
    loop {
        ls.service();
        // Scripted commands exercise the command path over the wire.
        if !queued_train && ls.sim_step >= 30 {
            if let Some(bi) = w.first_base(my_team) {
                ls.queue(world::Cmd::Train { building: w.ents[bi].id, unit: Kind::Worker });
                queued_train = true;
            }
        }
        if !queued_build && ls.sim_step >= 120 {
            let worker = w.ents.iter().find(|e| e.team == my_team && e.kind == Kind::Worker).map(|e| e.id);
            let base = w.ents.iter().find(|e| e.team == my_team && e.kind == Kind::Base).map(|e| e.pos);
            if let (Some(wid), Some(bp)) = (worker, base) {
                let site = if host { v2(bp.x + 130.0, bp.y - 130.0) } else { v2(bp.x - 130.0, bp.y + 130.0) };
                ls.queue(world::Cmd::Build { worker: wid, kind: Kind::Barracks, x: site.x, y: site.y, chain: false });
                queued_build = true;
            }
        }
        while ls.sim_step < max && ls.ready() {
            apply_step(&mut w, &ls.step_cmds());
            w.update(step);
            ls.advanced(w.checksum());
        }
        if ls.desync_step.is_some() || ls.sim_step >= max {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    match ls.desync_step {
        Some(s) => println!("{tag}: *** DESYNC at step {s} *** (last matched {})", ls.last_synced),
        None => println!(
            "{tag}: IN SYNC \u{2713}  ran {} steps, checksums matched through {}",
            ls.sim_step, ls.last_synced
        ),
    }
}

/// In-process 4-human-peer determinism test: a host plus three joiners over
/// loopback (exercising the star relay). All four must stay byte-identical.
fn mptest4() {
    use std::sync::mpsc;
    let port = 7831u16;
    let s = seed();
    let (tx, rx) = mpsc::channel();

    let tx0 = tx.clone();
    let host = std::thread::spawn(move || {
        let mut host = match net::Host::bind(port, s, 4) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("host bind failed: {e}");
                return;
            }
        };
        let mut spins = 0;
        while host.players() < 4 && spins < 5000 {
            host.poll();
            std::thread::sleep(std::time::Duration::from_millis(1));
            spins += 1;
        }
        if host.players() < 4 {
            eprintln!("host: not all joiners arrived ({}/4)", host.players());
            return;
        }
        let _ = tx0.send(run_peer(host.start()));
    });

    std::thread::sleep(std::time::Duration::from_millis(200));
    let mut handles = vec![host];
    for _ in 0..3 {
        let txj = tx.clone();
        handles.push(std::thread::spawn(move || {
            let mut j = match net::Joiner::connect(&format!("127.0.0.1:{port}")) {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("join failed: {e}");
                    return;
                }
            };
            let mut spins = 0;
            loop {
                if let Some(ls) = j.poll_start() {
                    let _ = txj.send(run_peer(ls));
                    return;
                }
                if spins > 8000 {
                    eprintln!("joiner: host never started");
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
                spins += 1;
            }
        }));
    }
    for h in handles {
        h.join().ok();
    }
    drop(tx);

    let mut results: Vec<(usize, Option<u32>, u32, u32, u64)> = rx.iter().collect();
    results.sort_by_key(|r| r.0);
    if results.len() != 4 {
        println!("MPTEST4: FAILED - only {}/4 peers finished", results.len());
        return;
    }
    for &(f, desync, synced, step, chk) in &results {
        println!("  faction {f}: reached step {step}  last_synced {synced}  desync {desync:?}  checksum {chk:016x}");
    }
    let chk0 = results[0].4;
    let in_sync = results.iter().all(|r| r.1.is_none() && r.3 == results[0].3 && r.4 == chk0);
    if in_sync {
        println!("MPTEST4: IN SYNC \u{2713} - all 4 human peers byte-identical through the relay");
    } else {
        println!("MPTEST4: *** DESYNC ***");
    }
}

/// Drive one peer's lockstep to a fixed step count, injecting one deterministic
/// command, and return (faction, desync, last_synced, final_step, checksum).
fn run_peer(mut ls: net::Lockstep) -> (usize, Option<u32>, u32, u32, u64) {
    let mut is_ai = [false; world::MAX_FACTIONS];
    for fi in 0..ls.factions {
        is_ai[fi] = ls.is_ai[fi];
    }
    let mut w = World::new_match(ls.seed, ls.factions, is_ai, Team::from_idx(ls.my_faction), true);
    let me = Team::from_idx(ls.my_faction);
    let max = 900u32;
    let mut injected = false;
    let mut idle_spins = 0;
    while ls.sim_step < max {
        if ls.sim_step == 30 && !injected {
            injected = true;
            let ids: Vec<u32> = w.ents.iter().filter(|e| e.team == me && e.kind == Kind::Worker).map(|e| e.id).collect();
            ls.queue(world::Cmd::Order { ids, x: w.world_w * 0.5, y: w.world_h * 0.5, attack_move: false, queue: false });
        }
        ls.service();
        if !ls.ready() {
            idle_spins += 1;
            if idle_spins > 20000 {
                break; // deadlock guard
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        }
        idle_spins = 0;
        apply_step(&mut w, &ls.step_cmds());
        w.update(1.0 / 60.0);
        ls.advanced(w.checksum());
        if ls.desync_step.is_some() {
            break;
        }
    }
    (ls.my_faction, ls.desync_step, ls.last_synced, ls.sim_step, w.checksum())
}

// ---- screen-region helpers ------------------------------------------------

/// Distance attenuation + stereo pan for a world sound given the view rect
/// `[vmin, vmax]`. Inside the view it's full volume; it fades to silence within
/// `FADE` world units beyond the edge. Pan tracks the source's horizontal
/// offset from the view centre.
fn sound_at_view(vmin: V2, vmax: V2, p: V2) -> (f32, f32) {
    const FADE: f32 = 360.0;
    let dx = (vmin.x - p.x).max(p.x - vmax.x).max(0.0);
    let dy = (vmin.y - p.y).max(p.y - vmax.y).max(0.0);
    let outside = dx.hypot(dy);
    let gain = (1.0 - outside / FADE).clamp(0.0, 1.0);
    let cx = (vmin.x + vmax.x) * 0.5;
    let half = ((vmax.x - vmin.x) * 0.5).max(1.0);
    let pan = ((p.x - cx) / half).clamp(-1.0, 1.0);
    (gain, pan)
}

fn in_hud(m: V2) -> bool {
    m.y < HUD_H as f32
}
fn in_minimap(m: V2, sw: i32, sh: i32) -> bool {
    let (mx, my) = (mm_x(sw), mm_y(sh));
    m.x >= mx as f32
        && m.x <= (mx + MM_W) as f32
        && m.y >= my as f32
        && m.y <= (my + MM_H) as f32
}
fn mm_to_world(m: V2, sw: i32, sh: i32, world_w: f32, world_h: f32) -> V2 {
    let (mx, my) = (mm_x(sw), mm_y(sh));
    v2(
        ((m.x - mx as f32) / MM_W as f32) * world_w,
        ((m.y - my as f32) / MM_H as f32) * world_h,
    )
}
/// Queue `unit` at every selected player building of `bkind`.
/// Queue a Train command for each selected building of `bkind`.
fn queue_train(world: &World, out: &mut Vec<world::Cmd>, bkind: Kind, unit: Kind) {
    // One unit per press, sent to the least-loaded selected building, so several
    // selected production buildings fill evenly rather than all at once.
    if let Some(id) = world.least_loaded_selected(bkind) {
        out.push(world::Cmd::Train { building: id, unit });
    }
}

fn has_player_worker(world: &World) -> bool {
    world
        .ents
        .iter()
        .any(|e| e.selected && e.team == world.my_team && e.kind == Kind::Worker)
}

fn first_selected_worker(world: &World) -> Option<u32> {
    world
        .ents
        .iter()
        .find(|e| e.selected && e.team == world.my_team && e.kind == Kind::Worker)
        .map(|e| e.id)
}

fn digit_of(k: Key) -> Option<usize> {
    match k {
        Key::Key0 => Some(0),
        Key::Key1 => Some(1),
        Key::Key2 => Some(2),
        Key::Key3 => Some(3),
        Key::Key4 => Some(4),
        Key::Key5 => Some(5),
        Key::Key6 => Some(6),
        Key::Key7 => Some(7),
        Key::Key8 => Some(8),
        Key::Key9 => Some(9),
        _ => None,
    }
}

// ---- colors ---------------------------------------------------------------

fn fill_of(team: Team) -> u32 {
    match team {
        Team::Player => rgb(64, 132, 238),   // blue
        Team::Enemy => rgb(228, 72, 60),     // red
        Team::Faction2 => rgb(86, 196, 110), // green
        Team::Faction3 => rgb(208, 150, 56), // amber
        Team::Neutral => rgb(120, 124, 134),
    }
}
fn edge_of(team: Team) -> u32 {
    match team {
        Team::Player => rgb(150, 200, 255),
        Team::Enemy => rgb(255, 150, 140),
        Team::Faction2 => rgb(170, 240, 185),
        Team::Faction3 => rgb(250, 210, 140),
        Team::Neutral => rgb(180, 184, 196),
    }
}

/// Short human label for a faction (for the HUD / messages).
fn faction_name(team: Team) -> &'static str {
    match team {
        Team::Player => "BLUE",
        Team::Enemy => "RED",
        Team::Faction2 => "GREEN",
        Team::Faction3 => "AMBER",
        Team::Neutral => "NEUTRAL",
    }
}

const SELECT: u32 = 0xFF6CFF8C;
const MINERAL: u32 = 0xFF5AD2DC;
const INK: u32 = 0xFF0A0C10;

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Base => "COMMAND CENTER",
        Kind::Barracks => "BARRACKS",
        Kind::Factory => "FACTORY",
        Kind::Depot => "SUPPLY DEPOT",
        Kind::Worker => "WORKER",
        Kind::Soldier => "SOLDIER",
        Kind::Tank => "TANK",
        Kind::Pyro => "PYRO",
        Kind::Raider => "RAIDER",
        Kind::Mortar => "MORTAR",
        Kind::Sapper => "SAPPER",
        Kind::Mineral => "MINERAL",
    }
}

// ---- rendering ------------------------------------------------------------

fn render(c: &mut Canvas, w: &World, ui: &Ui, mouse: V2) {
    // Screen shake offsets the whole world view (but not the HUD/minimap).
    let camx = (w.cam.x + w.shake_off.x) as i32;
    let camy = (w.cam.y + w.shake_off.y) as i32;
    let w2s = |p: V2| -> (i32, i32) { (p.x as i32 - camx, p.y as i32 - camy) };

    // Ground.
    c.clear(rgb(22, 25, 20));
    draw_grid(c, camx, camy);
    draw_terrain(c, w, camx, camy);

    // Map border (uses the actual world size — it grows with the player count).
    let (bx, by) = w2s(v2(0.0, 0.0));
    c.rect(bx, by, w.world_w as i32, w.world_h as i32, rgb(46, 52, 42));

    // Minerals.
    for e in &w.ents {
        if e.kind != Kind::Mineral {
            continue;
        }
        let (sx, sy) = w2s(e.pos);
        if sx < -30 || sy < -30 || sx > c.w + 30 || sy > c.h + 30 {
            continue;
        }
        // A patch shrinks and dims as it's mined out, so its remaining wealth
        // reads at a glance.
        let frac = (e.minerals as f32 / world::MINERAL_START as f32).clamp(0.0, 1.0);
        let outer = 6 + (7.0 * frac) as i32;
        let inner = 4 + (5.0 * frac) as i32;
        let body = gfx::mix(rgb(34, 86, 96), MINERAL, 0.45 + 0.55 * frac);
        c.fill_diamond(sx, sy, outer, rgb(28, 120, 130));
        c.fill_diamond(sx, sy, inner, body);
        c.fill_diamond(sx - 2, sy - 2, (inner / 3).max(1), rgb(190, 250, 255));
    }

    // Pending build ghosts (player workers walking out to construct).
    for e in &w.ents {
        if e.team != Team::Player {
            continue;
        }
        if let Order::Build(k, site) = e.order {
            let (sx, sy) = w2s(site);
            let r = radius(k) as i32;
            c.rect(sx - r, sy - r, r * 2, r * 2, rgb(90, 150, 90));
            c.dashed_line(sx - r, sy - r, sx + r, sy - r, rgb(120, 200, 120), 3);
        }
    }

    // Buildings (enemy ones only while in sight).
    for e in &w.ents {
        if !world::is_building(e.kind) {
            continue;
        }
        if e.team != w.my_team && w.vis_at(e.pos) != 2 {
            continue;
        }
        draw_building(c, e, w2s(e.pos));
    }

    // Rally lines for selected production buildings (depots train nothing, so
    // they have no rally to show).
    for e in &w.ents {
        if e.selected && matches!(e.kind, Kind::Base | Kind::Barracks | Kind::Factory) {
            let (sx, sy) = w2s(e.pos);
            let (rx, ry) = w2s(e.rally);
            c.dashed_line(sx, sy, rx, ry, rgb(120, 220, 140), 4);
            c.fill_diamond(rx, ry, 4, SELECT);
        }
    }

    // Units (enemy ones only while in sight).
    for e in &w.ents {
        if !(world::is_army(e.kind) || e.kind == Kind::Worker) {
            continue;
        }
        if e.team != w.my_team && w.vis_at(e.pos) != 2 {
            continue;
        }
        let (sx, sy) = w2s(e.pos);
        if sx < -20 || sy < -20 || sx > c.w + 20 || sy > c.h + 20 {
            continue;
        }
        let mut fill = fill_of(e.team);
        if e.flash > 0.0 {
            fill = gfx::mix(fill, rgb(255, 255, 255), 0.7);
        }
        let r = radius(e.kind) as i32;
        if e.selected {
            c.circle(sx, sy, r + 4, SELECT);
            c.circle(sx, sy, r + 5, SELECT);
        }
        // Shared retro treatment: a bright phosphor outline, a saturated body, a
        // near-white highlight, and a glowing accent (engine / muzzle / sensor).
        let f = e.facing;
        let (px, py) = (-f.y, f.x);
        let rf = r as f32;
        let ec = edge_of(e.team);
        let hi = gfx::mix(fill, rgb(255, 255, 255), 0.55);
        let dk = gfx::mix(fill, rgb(0, 0, 0), 0.5);
        // A point `fwd` along the heading and `side` across it, in screen space.
        let pt = |fwd: f32, side: f32| -> (i32, i32) {
            ((sx as f32 + f.x * fwd + px * side) as i32, (sy as f32 + f.y * fwd + py * side) as i32)
        };
        // Faint glow halo so every unit reads as a lit object on the dark field.
        c.fill_circle_add(sx, sy, r, ec, 0.10);
        match e.kind {
            Kind::Worker => {
                // Round utility drone with a glowing forward sensor.
                c.fill_circle(sx, sy, r, ec);
                c.fill_circle(sx, sy, r - 1, fill);
                c.fill_circle(sx - 1, sy - 1, (r - 4).max(1), hi); // top-left glint
                let (ex, ey) = pt(rf - 1.0, 0.0);
                c.fill_circle_add(ex, ey, 2, rgb(150, 245, 255), 0.9);
                if e.carry > 0 {
                    let (cx2, cy2) = pt(-rf - 1.0, 0.0);
                    c.fill_diamond(cx2, cy2, 3, MINERAL);
                    c.fill_diamond(cx2, cy2, 1, rgb(210, 250, 255));
                }
            }
            Kind::Tank => {
                // Treaded hull + a rotating turret with a long barrel.
                let tl = rf + 2.0;
                for s in [-1.0f32, 1.0] {
                    let (tx, ty) = pt(0.0, s * rf * 0.72);
                    c.fill_orect(tx, ty, tl, 2.2, f, dk); // tread rail
                }
                c.fill_orect(sx, sy, rf, rf * 0.72, f, ec); // hull rim
                c.fill_orect(sx, sy, rf - 1.5, rf * 0.72 - 1.5, f, fill);
                c.fill_orect(sx, sy, rf * 0.5, rf * 0.4, f, hi); // glacis plate
                let bl = rf + 8.0;
                let (bmx, bmy) = pt(bl * 0.55, 0.0);
                c.fill_orect(bmx, bmy, bl * 0.55, 1.7, f, ec); // barrel
                c.fill_circle(sx, sy, 4, ec); // turret
                c.fill_circle(sx, sy, 2, hi);
                let (mzx, mzy) = pt(bl + 1.0, 0.0);
                c.fill_circle_add(mzx, mzy, 2, rgb(255, 210, 120), 0.45); // muzzle ember
            }
            Kind::Raider => {
                // A pointed dart: bright-edged arrowhead, a canopy stripe, exhaust.
                let nose = pt(rf * 2.0, 0.0);
                let bl = pt(-rf * 0.5, rf * 0.76);
                let br = pt(-rf * 0.5, -rf * 0.76);
                c.fill_tri(nose.0, nose.1, bl.0, bl.1, br.0, br.1, ec); // bright outline
                let inose = pt(rf * 1.4, 0.0);
                let ibl = pt(-rf * 0.15, rf * 0.5);
                let ibr = pt(-rf * 0.15, -rf * 0.5);
                c.fill_tri(inose.0, inose.1, ibl.0, ibl.1, ibr.0, ibr.1, fill); // body
                let (cmx, cmy) = pt(rf * 0.35, 0.0);
                c.fill_orect(cmx, cmy, rf * 0.55, 1.1, f, hi); // canopy stripe
                let (gx, gy) = pt(-rf * 0.7, 0.0);
                c.fill_circle_add(gx, gy, 3, rgb(130, 205, 255), 0.6); // exhaust
            }
            Kind::Pyro => {
                // Diamond trooper with a fuel pack and a live pilot flame.
                let (fx, fy) = pt(-rf * 0.85, 0.0);
                c.fill_orect(fx, fy, rf * 0.5, rf * 0.7, f, dk); // fuel pack
                c.fill_diamond(sx, sy, r, ec);
                c.fill_diamond(sx, sy, r - 2, fill);
                c.fill_diamond(sx - 1, sy - 1, (r - 4).max(1), hi);
                let (nx, ny) = pt(rf * 0.9, 0.0);
                c.fill_orect(nx, ny, rf * 0.7, 1.4, f, ec); // nozzle
                let (tx, ty) = pt(rf + 5.0, 0.0);
                let flick = 0.6 + 0.4 * ((w.time * 19.0).sin() * 0.5 + 0.5);
                c.fill_circle_add(tx, ty, 3, rgb(255, 130, 40), flick);
                c.fill_circle_add(tx, ty, 1, rgb(255, 240, 180), flick);
            }
            Kind::Mortar => {
                // Boxy tracked hull with a short, thick mortar tube on a baseplate
                // — clearly a siege piece, not the Tank's long turret gun.
                for s in [-1.0f32, 1.0] {
                    let (tx, ty) = pt(0.0, s * rf * 0.78);
                    c.fill_orect(tx, ty, rf * 0.95, 2.4, f, dk); // tread rails
                }
                c.fill_orect(sx, sy, rf * 0.92, rf * 0.82, f, ec); // hull
                c.fill_orect(sx, sy, rf * 0.92 - 1.5, rf * 0.82 - 1.5, f, fill);
                c.fill_circle(sx, sy, (rf * 0.42) as i32, dk); // baseplate
                let (bmx, bmy) = pt(rf * 0.95, 0.0); // stubby thick tube
                c.fill_orect(bmx, bmy, rf * 0.85, 2.9, f, hi);
                let (mz, mzy) = pt(rf * 1.85, 0.0);
                c.fill_circle(mz, mzy, 2, dk); // muzzle mouth
                c.fill_circle_add(mz, mzy, 2, rgb(255, 205, 130), 0.35);
            }
            Kind::Sapper => {
                // A charger hauling a live charge: round body, a fat dark bomb pack
                // at the rear, and a blinking red arming light.
                let (bx, by) = pt(-1.0, 0.0);
                c.fill_circle(bx, by, r, ec);
                c.fill_circle(bx, by, r - 1, fill);
                let (cx, cy) = pt(-rf * 0.9, 0.0);
                c.fill_circle(cx, cy, (r - 2).max(2), dk);
                c.fill_circle(cx, cy, (r - 4).max(1), gfx::mix(dk, rgb(255, 90, 60), 0.45));
                let (hx, hy) = pt(rf * 0.7, 0.0);
                c.fill_circle(hx, hy, (r - 4).max(1), hi); // leaning head
                let blink = 0.35 + 0.65 * ((w.time * 9.0).sin() * 0.5 + 0.5);
                c.fill_circle_add(bx, by - 1, 2, rgb(255, 70, 50), blink);
                c.fill_circle(bx, by - 1, 1, rgb(255, 235, 225));
            }
            _ => {
                // Soldier: a trooper body with a bright helmet and a rifle that
                // clearly leads the heading.
                let (bx, by) = pt(-1.5, 0.0);
                c.fill_circle(bx, by, r, ec);
                c.fill_circle(bx, by, r - 1, fill);
                // Rifle: a bright bar reaching well past the body on one shoulder.
                let (gmx, gmy) = pt(rf * 0.9, 2.2);
                c.fill_orect(gmx, gmy, rf * 1.15, 1.3, f, gfx::mix(ec, rgb(255, 255, 255), 0.35));
                let (mtx, mty) = pt(rf * 2.1, 2.2);
                c.fill_circle_add(mtx, mty, 1, rgb(255, 225, 150), 0.6); // muzzle hint
                // Helmet: a bright dot leading the body.
                let (hx, hy) = pt(1.5, -0.5);
                c.fill_circle(hx, hy, (r - 4).max(1), hi);
            }
        }
        if e.hp < e.max_hp {
            draw_hpbar(c, sx, sy - r - 6, (r * 2 + 4).max(12), e.hp / e.max_hp, e.team);
        }
    }

    // Tracers (weapon fire).
    for t in &w.tracers {
        let (ax, ay) = w2s(t.a);
        let (bx2, by2) = w2s(t.b);
        let a = (t.life / 0.08).clamp(0.0, 1.0);
        let col = gfx::mix(rgb(22, 25, 20), t.color, a);
        c.line(ax, ay, bx2, by2, col);
    }

    // Shockwave rings (drawn under particles), expanding and fading.
    for s in &w.shocks {
        let (px, py) = w2s(s.pos);
        let prog = 1.0 - (s.life / s.max_life).clamp(0.0, 1.0); // 0 -> 1
        let rr = (s.max_r * (1.0 - (1.0 - prog).powi(2))) as i32; // ease-out
        let intensity = (s.life / s.max_life).clamp(0.0, 1.0) * 0.9;
        c.ring_add(px, py, (rr - 2).max(0), rr + 1, s.color, intensity);
    }

    // Particles: explosions, sparks, smoke, sparkle. Glow ones blow out white.
    for p in &w.particles {
        let (px, py) = w2s(p.pos);
        if px < -8 || py < -8 || px > c.w + 8 || py > c.h + 8 {
            continue;
        }
        let t = (p.life / p.max_life).clamp(0.0, 1.0); // 1 at birth -> 0 at death
        let r = (p.size * (0.35 + 0.65 * t)) as i32;
        if p.glow {
            // Additive: bright early, fading; overlaps stack toward white.
            let intensity = (t * t * 1.3).min(1.0);
            c.fill_circle_add(px, py, r.max(1), p.color, intensity);
        } else {
            let col = gfx::mix(rgb(22, 25, 20), p.color, t);
            if r <= 1 {
                c.put(px, py, col);
                c.put(px + 1, py, col);
            } else {
                c.fill_circle(px, py, r, col);
            }
        }
    }

    // Command feedback pings: a ring that expands and fades at the target.
    for &(p, life, col) in &w.pings {
        let (px, py) = w2s(p);
        let t = (1.0 - (life / 0.5)).clamp(0.0, 1.0);
        let r = 4 + (t * 14.0) as i32;
        let fade = gfx::mix(rgb(22, 25, 20), col, (life / 0.5).clamp(0.0, 1.0));
        c.circle(px, py, r, fade);
        c.circle(px, py, r + 1, fade);
    }

    // Queued-order waypoints: trace the path of any selected unit that has a
    // chain of orders, so Shift-queued moves are visible.
    let order_pt = |o: &Order| -> Option<V2> {
        match o {
            Order::Move(p) | Order::AttackMove(p) | Order::Build(_, p) => Some(*p),
            Order::Attack(id) | Order::Gather(id) | Order::Repair(id) => {
                w.index_of(*id).map(|j| w.ents[j].pos)
            }
            Order::Idle => None,
        }
    };
    for e in &w.ents {
        if !e.selected || e.team != w.my_team || e.order_queue.is_empty() {
            continue;
        }
        let mut prev = w2s(e.pos);
        let line_col = rgb(120, 200, 150);
        for o in std::iter::once(&e.order).chain(e.order_queue.iter()) {
            if let Some(p) = order_pt(o) {
                let s = w2s(p);
                c.dashed_line(prev.0, prev.1, s.0, s.1, line_col, 4);
                c.fill_circle(s.0, s.1, 2, line_col);
                prev = s;
            }
        }
    }

    // ---- Fog of war: black out the unseen, dim what's only remembered. ----
    let cs = w.fog_cell as i32;
    let gx0 = (camx / cs).max(0);
    let gy0 = (camy / cs).max(0);
    let gx1 = (((camx + c.w) / cs) + 1).min(w.fog_w as i32 - 1);
    let gy1 = (((camy + c.h) / cs) + 1).min(w.fog_h as i32 - 1);
    let unseen = rgb(6, 7, 10);
    for gy in gy0..=gy1 {
        let row = gy as usize * w.fog_w;
        for gx in gx0..=gx1 {
            let v = w.vis[row + gx as usize];
            if v == 2 {
                continue;
            }
            let sx = gx * cs - camx;
            let sy = gy * cs - camy;
            if v == 0 {
                c.fill_rect(sx, sy, cs, cs, unseen);
            } else {
                c.fill_rect_a(sx, sy, cs, cs, unseen, 0.55);
            }
        }
    }

    // Selection drag box.
    if ui.dragging {
        if let Some(ds) = ui.drag_start {
            let (ax, ay) = w2s(ds);
            let (bx2, by2) = (mouse.x as i32, mouse.y as i32);
            let x0 = ax.min(bx2);
            let y0 = ay.min(by2);
            let bw = (ax - bx2).abs();
            let bh = (ay - by2).abs();
            c.fill_rect_a(x0, y0, bw, bh, rgb(110, 255, 150), 0.10);
            c.rect(x0, y0, bw, bh, SELECT);
        }
    }

    // Build placement ghost following the cursor.
    if let Some(k) = ui.build_mode {
        // Show already-chained build sites for the selected workers so the player
        // sees what Shift-clicking has queued up.
        let pending = rgb(120, 200, 235);
        for e in &w.ents {
            if e.kind != Kind::Worker || !e.selected || e.team != w.my_team {
                continue;
            }
            let mut mark = |bk: Kind, site: V2| {
                let (px, py) = w2s(site);
                let rr = radius(bk) as i32;
                c.fill_rect_a(px - rr, py - rr, rr * 2, rr * 2, pending, 0.18);
                c.rect(px - rr, py - rr, rr * 2, rr * 2, pending);
            };
            if let Order::Build(bk, site) = e.order {
                mark(bk, site);
            }
            for &(bk, site) in &e.build_queue {
                mark(bk, site);
            }
        }
        let wm = v2(mouse.x + w.cam.x, mouse.y + w.cam.y);
        let ok = w.can_build(k, wm) && w.team_min(w.my_team) >= cost(k);
        let col = if ok { rgb(110, 230, 120) } else { rgb(230, 90, 80) };
        let r = radius(k) as i32;
        let (sx, sy) = (mouse.x as i32, mouse.y as i32);
        // Required clear radius from other buildings — drawn so the spacing rule
        // is visible while placing.
        let clear_r = (radius(k) + world::BUILD_GAP) as i32;
        c.circle(sx, sy, clear_r, gfx::mix(rgb(22, 25, 20), col, 0.55));
        c.fill_rect_a(sx - r, sy - r, r * 2, r * 2, col, 0.25);
        c.rect(sx - r, sy - r, r * 2, r * 2, col);
        // If a nearby building is what's blocking the spot, ring it in red so the
        // reason is obvious.
        if !ok {
            for e in &w.ents {
                if !world::is_building(e.kind) {
                    continue;
                }
                let need = radius(k) + radius(e.kind) + world::BUILD_GAP;
                if e.pos.dist(wm) < need + 2.0 {
                    let (ex, ey) = w2s(e.pos);
                    c.circle(ex, ey, (radius(e.kind) + world::BUILD_GAP) as i32, rgb(235, 95, 85));
                }
            }
        }
    }

    // Full-screen flash from big explosions (over the world, under the HUD).
    if w.flash_amt > 0.01 {
        c.fill_rect_a(0, 0, c.w, c.h, w.flash_color, w.flash_amt.min(0.85));
    }

    draw_minimap(c, w);
    draw_hud(c, w, ui);
    if ui.show_help {
        draw_help(c);
    }
    if w.over != 0 {
        draw_gameover(c, w, mouse);
    }
    if ui.menu {
        draw_menu(c, mouse);
    }
}

fn draw_menu(c: &mut Canvas, mouse: V2) {
    c.fill_rect_a(0, 0, c.w, c.h, INK, 0.72);
    c.text_center(c.w / 2, c.h / 2 - 150, "PAUSED", rgb(255, 240, 200), 6);
    for (b, _) in pause_buttons(c) {
        let hover = b.hit(mouse);
        b.draw(c, hover);
    }
}

/// Draw the terrain: lighter raised plateaus rimmed with sunlit top edges, dark
/// impassable cliffs, and hatched ramps. Drawn over the grid, under everything.
fn draw_terrain(c: &mut Canvas, w: &World, camx: i32, camy: i32) {
    if !w.has_cliffs {
        return;
    }
    let cs = world::TCELL as i32;
    let tx0 = (camx / cs).max(0);
    let ty0 = (camy / cs).max(0);
    let tx1 = (((camx + c.w) / cs) + 1).min(w.tw as i32 - 1);
    let ty1 = (((camy + c.h) / cs) + 1).min(w.th as i32 - 1);
    let high = rgb(54, 62, 44);
    let high_rim = rgb(96, 112, 76);
    let high_shadow = rgb(34, 39, 29);
    // Cliffs are a cool blue-grey *rock*, deliberately lighter and bluer than the
    // near-black fog (rgb 6,7,10) so a visible cliff never reads as unexplored.
    let cliff = rgb(40, 44, 53);
    let cliff_top = rgb(92, 99, 113);
    let cliff_base = rgb(23, 25, 32);
    let cliff_seam = rgb(30, 33, 41);
    let ramp = rgb(58, 52, 34);
    let ramp_line = rgb(92, 84, 56);
    let tile = |tx: i32, ty: i32| -> u8 {
        if tx < 0 || ty < 0 || tx >= w.tw as i32 || ty >= w.th as i32 {
            world::T_OPEN
        } else {
            w.terrain[ty as usize * w.tw + tx as usize]
        }
    };
    for ty in ty0..=ty1 {
        for tx in tx0..=tx1 {
            let code = tile(tx, ty);
            if code == world::T_OPEN {
                continue;
            }
            let sx = tx * cs - camx;
            let sy = ty * cs - camy;
            match code {
                world::T_HIGH => {
                    c.fill_rect(sx, sy, cs, cs, high);
                    // Sunlit rim to the north, shadow drop to the south — a lit plateau.
                    if tile(tx, ty - 1) != world::T_HIGH {
                        c.fill_rect(sx, sy, cs, 2, high_rim);
                    }
                    if tile(tx, ty + 1) != world::T_HIGH {
                        c.fill_rect(sx, sy + cs - 2, cs, 2, high_shadow);
                    }
                }
                world::T_CLIFF => {
                    c.fill_rect(sx, sy, cs, cs, cliff);
                    // Lit cap on top, shadow at the foot, and a couple of vertical
                    // seams — reads as a solid raised rock wall, not void.
                    c.fill_rect(sx, sy, cs, 3, cliff_top);
                    c.fill_rect(sx, sy + cs - 3, cs, 3, cliff_base);
                    c.fill_rect(sx + cs / 3, sy + 4, 1, cs - 8, cliff_seam);
                    c.fill_rect(sx + 2 * cs / 3, sy + 4, 1, cs - 8, cliff_seam);
                }
                world::T_RAMP => {
                    c.fill_rect(sx, sy, cs, cs, ramp);
                    for k in 0..3 {
                        let o = k * 20 + 8;
                        c.line(sx + o, sy + cs - 2, sx + o + 14, sy + 2, ramp_line);
                    }
                }
                _ => {}
            }
        }
    }
}

fn draw_grid(c: &mut Canvas, camx: i32, camy: i32) {
    let step = 128;
    let line = rgb(28, 32, 26);
    let mut gx = -(camx % step);
    while gx < c.w {
        if gx >= 0 {
            c.fill_rect(gx, 0, 1, c.h, line);
        }
        gx += step;
    }
    let mut gy = -(camy % step);
    while gy < c.h {
        if gy >= 0 {
            c.fill_rect(0, gy, c.w, 1, line);
        }
        gy += step;
    }
}

fn draw_building(c: &mut Canvas, e: &world::Ent, (sx, sy): (i32, i32)) {
    let r = radius(e.kind) as i32;
    let mut fill = fill_of(e.team);
    if e.flash > 0.0 {
        fill = gfx::mix(fill, rgb(255, 255, 255), 0.6);
    }
    let ec = edge_of(e.team);
    let hi = gfx::mix(fill, rgb(255, 255, 255), 0.5);
    let dk = gfx::mix(fill, rgb(0, 0, 0), 0.45);
    let hole = gfx::mix(fill, rgb(0, 0, 0), 0.72); // dark doors / windows
    let warm = rgb(255, 200, 110);

    if e.selected {
        c.rect(sx - r - 4, sy - r - 4, (r + 4) * 2, (r + 4) * 2, SELECT);
    }
    // Soft footing glow so a building reads as a lit structure, not a flat tile.
    c.fill_circle_add(sx, sy, r + 2, ec, 0.07);

    // Beveled panel: shadow edge, body, lit top strip, shadowed base, team frame.
    c.fill_rect(sx - r, sy - r, r * 2, r * 2, dk);
    c.fill_rect(sx - r + 1, sy - r + 1, r * 2 - 2, r * 2 - 2, fill);
    c.fill_rect(sx - r + 1, sy - r + 1, r * 2 - 2, 2, hi);
    c.fill_rect(sx - r + 1, sy + r - 3, r * 2 - 2, 2, dk);
    c.rect(sx - r, sy - r, r * 2, r * 2, ec);

    match e.kind {
        Kind::Base => {
            // Fortified command hub: corner pylons around a glowing reactor core.
            for &(ox, oy) in &[(-1i32, -1i32), (1, -1), (-1, 1), (1, 1)] {
                let (cx, cy) = (sx + ox * (r - 4), sy + oy * (r - 4));
                c.fill_rect(cx - 3, cy - 3, 6, 6, ec);
                c.fill_rect(cx - 2, cy - 2, 4, 4, hi);
            }
            let cr = r / 3 + 1;
            c.fill_circle(sx, sy, cr, dk);
            c.fill_circle(sx, sy, cr - 2, ec);
            c.fill_circle_add(sx, sy, cr, rgb(150, 220, 255), 0.30);
            c.fill_diamond(sx, sy, 3, rgb(225, 245, 255));
        }
        Kind::Barracks => {
            // Infantry hall: roof ridge, a bay door, lit windows, a beacon antenna.
            c.fill_rect(sx - r + 4, sy - r + 4, r * 2 - 8, 2, hi);
            let dw = (r * 2) / 3;
            c.fill_rect(sx - dw / 2, sy + 1, dw, r - 1, hole);
            c.fill_rect(sx - dw / 2 + 1, sy + 2, dw - 2, r - 3, gfx::mix(hole, ec, 0.18));
            for k in 0..2 {
                c.fill_rect(sx - r + 5 + k * 6, sy - r + 8, 3, 4, gfx::mix(fill, ec, 0.5));
            }
            c.line(sx + r - 7, sy - r, sx + r - 7, sy - r - 6, ec);
            c.fill_circle_add(sx + r - 7, sy - r - 6, 2, ec, 0.85);
        }
        Kind::Factory => {
            // Vehicle works: a wide slatted bay door and a smoking stack.
            let bw = r * 2 - 8;
            c.fill_rect(sx - r + 4, sy, bw, r - 3, hole);
            for k in 0..4 {
                c.fill_rect(sx - r + 5 + k * (bw / 4), sy + 1, 1, r - 5, dk);
            }
            c.fill_rect(sx - r + 4, sy - 2, bw, 2, ec);
            let stx = sx - r + 7;
            c.fill_rect(stx - 2, sy - r - 6, 5, 8, dk);
            c.fill_rect(stx - 1, sy - r - 5, 3, 7, gfx::mix(dk, ec, 0.4));
            c.fill_circle_add(stx, sy - r - 7, 3, warm, 0.45);
        }
        Kind::Depot => {
            // Stacked supply crates with a status light.
            for &(ox, oy) in &[(-1i32, -1i32), (1, -1), (-1, 1), (1, 1)] {
                let (cx, cy) = (sx + ox * (r / 2), sy + oy * (r / 2));
                let q = r - 2;
                c.fill_rect(cx - q / 2, cy - q / 2, q, q, dk);
                c.fill_rect(cx - q / 2 + 1, cy - q / 2 + 1, q - 2, q - 2, fill);
                c.fill_rect(cx - q / 2 + 1, cy - q / 2 + 1, q - 2, 1, hi);
            }
            c.fill_circle_add(sx, sy, 3, rgb(150, 255, 180), 0.5);
            c.fill_circle(sx, sy, 1, rgb(210, 255, 220));
        }
        _ => {}
    }

    // Under construction: the structure rises out of a dark foundation from the
    // bottom up, a glowing beam sweeping along the build front. (The detailed art
    // above is drawn first; here we mask off the part not yet raised.)
    if e.build_left > 0.0 {
        let bt = world::build_time(e.kind).max(0.001);
        let frac = (1.0 - e.build_left / bt).clamp(0.0, 1.0);
        let front = sy + r - (2.0 * r as f32 * frac) as i32; // y of the build line
        let foundation = gfx::mix(fill, rgb(0, 0, 0), 0.82);
        let scaf = gfx::mix(ec, warm, 0.45);
        let cover_h = front - (sy - r);
        if cover_h > 0 {
            c.fill_rect(sx - r, sy - r, r * 2, cover_h, foundation);
            // Scaffold: corner uprights and faint rungs in the unbuilt span.
            c.fill_rect(sx - r, sy - r, 2, cover_h, scaf);
            c.fill_rect(sx + r - 2, sy - r, 2, cover_h, scaf);
            let mut yy = sy - r + 5;
            while yy < front {
                c.fill_rect(sx - r, yy, r * 2, 1, gfx::mix(foundation, scaf, 0.45));
                yy += 9;
            }
        }
        // Glowing build beam at the front.
        c.fill_rect_a(sx - r, front - 3, r * 2, 6, warm, 0.30);
        c.fill_rect(sx - r, front - 1, r * 2, 2, gfx::mix(warm, rgb(255, 255, 255), 0.5));
        c.rect(sx - r, sy - r, r * 2, r * 2, scaf);
    }

    // HP bar.
    draw_hpbar(c, sx, sy - r - 8, r * 2, e.hp / e.max_hp, e.team);

    // Production progress + queue size.
    if !e.queue.is_empty() {
        let total = world::build_time(e.queue[0]).max(0.001);
        let frac = (e.train_timer / total).clamp(0.0, 1.0);
        let bw = r * 2;
        c.fill_rect(sx - r, sy + r + 4, bw, 4, rgb(20, 24, 30));
        c.fill_rect(sx - r, sy + r + 4, (bw as f32 * frac) as i32, 4, rgb(120, 220, 250));
        if e.queue.len() > 1 {
            c.text(sx + r + 2, sy + r, &format!("{}", e.queue.len()), rgb(220, 220, 230), 1);
        }
    }
}

fn draw_hpbar(c: &mut Canvas, cx: i32, top: i32, width: i32, frac: f32, team: Team) {
    let frac = frac.clamp(0.0, 1.0);
    let x = cx - width / 2;
    c.fill_rect(x - 1, top - 1, width + 2, 5, rgb(8, 10, 12));
    let fg = if team == Team::Player {
        if frac > 0.5 {
            rgb(90, 220, 110)
        } else if frac > 0.25 {
            rgb(235, 200, 70)
        } else {
            rgb(230, 80, 70)
        }
    } else {
        rgb(228, 90, 80)
    };
    c.fill_rect(x, top, (width as f32 * frac) as i32, 3, fg);
}

fn draw_minimap(c: &mut Canvas, w: &World) {
    let (mmx, mmy) = (mm_x(c.w), mm_y(c.h));
    c.fill_rect_a(mmx - 4, mmy - 4, MM_W + 8, MM_H + 8, INK, 0.85);
    c.rect(mmx - 4, mmy - 4, MM_W + 8, MM_H + 8, rgb(60, 66, 56));
    c.fill_rect(mmx, mmy, MM_W, MM_H, rgb(16, 18, 14));

    // Scale to the ACTUAL world size — maps grow with the player count, so using
    // the fixed WORLD_W/WORLD_H constants made 3-4p minimap content overflow the
    // frame and spill into the bottom bar.
    let sx = MM_W as f32 / w.world_w;
    let sy = MM_H as f32 / w.world_h;
    // Terrain underlay: high ground lighter, cliffs darker.
    if w.has_cliffs {
        for ty in 0..w.th {
            for tx in 0..w.tw {
                let code = w.terrain[ty * w.tw + tx];
                let col = match code {
                    world::T_HIGH => rgb(40, 48, 36),
                    world::T_CLIFF => rgb(8, 9, 8),
                    world::T_RAMP => rgb(44, 42, 31),
                    _ => continue,
                };
                let bx = mmx + (tx as f32 * world::TCELL * sx) as i32;
                let by = mmy + (ty as f32 * world::TCELL * sy) as i32;
                let bw = (world::TCELL * sx).ceil() as i32;
                let bh = (world::TCELL * sy).ceil() as i32;
                c.fill_rect(bx, by, bw, bh, col);
            }
        }
    }
    for e in &w.ents {
        // Don't reveal enemies the player can't currently see.
        if e.team != w.my_team && w.vis_at(e.pos) != 2 {
            continue;
        }
        // Minerals only show once their patch has been explored.
        if e.kind == Kind::Mineral && w.vis_at(e.pos) == 0 {
            continue;
        }
        let bx = mmx + (e.pos.x * sx) as i32;
        let by = mmy + (e.pos.y * sy) as i32;
        let (col, sz) = match e.kind {
            Kind::Mineral => (MINERAL, 1),
            Kind::Base => (edge_of(e.team), 3),
            Kind::Barracks | Kind::Factory | Kind::Depot => (edge_of(e.team), 2),
            Kind::Tank | Kind::Mortar => (fill_of(e.team), 2),
            _ => (fill_of(e.team), 1),
        };
        c.fill_rect(bx - sz / 2, by - sz / 2, sz.max(1) + 1, sz.max(1) + 1, col);
    }

    // Fog of war on the minimap: black out what you've never seen, and dim the
    // ground you've explored but can't currently see — so the map fills in as you
    // scout and you can read where you've already been.
    let cells = w.fog_w * w.fog_h;
    let base = w.my_team.idx() * cells;
    let unseen = rgb(11, 12, 15);
    for fy in 0..w.fog_h {
        for fx in 0..w.fog_w {
            let v = w.vis[base + fy * w.fog_w + fx];
            if v == 2 {
                continue;
            }
            // Tile each cell exactly up to the next so the dim layer never overlaps
            // itself (which would blotch the alpha).
            let bx0 = mmx + (fx as f32 * w.fog_cell * sx) as i32;
            let bx1 = mmx + ((fx + 1) as f32 * w.fog_cell * sx) as i32;
            let by0 = mmy + (fy as f32 * w.fog_cell * sy) as i32;
            let by1 = mmy + ((fy + 1) as f32 * w.fog_cell * sy) as i32;
            let (bw, bh) = ((bx1 - bx0).max(1), (by1 - by0).max(1));
            if v == 0 {
                c.fill_rect(bx0, by0, bw, bh, unseen); // never seen — solid
            } else {
                c.fill_rect_a(bx0, by0, bw, bh, unseen, 0.5); // remembered — dimmed
            }
        }
    }

    // Camera viewport.
    let vx = mmx + (w.cam.x * sx) as i32;
    let vy = mmy + (w.cam.y * sy) as i32;
    let vw = (c.w as f32 * sx) as i32;
    let vh = (c.h as f32 * sy) as i32;
    c.rect(vx, vy, vw, vh, rgb(230, 230, 240));

    // Echo command pings on the minimap, so orders you issue *from* the minimap
    // (a move, or an attack onto a spotted enemy) flash right where you clicked.
    for &(p, life, col) in &w.pings {
        let px = mmx + (p.x * sx) as i32;
        let py = mmy + (p.y * sy) as i32;
        if px < mmx || px > mmx + MM_W || py < mmy || py > mmy + MM_H {
            continue;
        }
        let t = (1.0 - life / 0.5).clamp(0.0, 1.0);
        let fade = gfx::mix(rgb(16, 18, 14), col, (life / 0.5).clamp(0.0, 1.0));
        c.circle(px, py, 2 + (t * 8.0) as i32, fade);
        c.circle(px, py, 3 + (t * 8.0) as i32, fade);
        c.fill_rect(px - 1, py - 1, 3, 3, col); // bright core, easy to spot
    }
}

fn draw_hud(c: &mut Canvas, w: &World, ui: &Ui) {
    c.fill_rect_a(0, 0, c.w, HUD_H, INK, 0.88);
    c.fill_rect(0, HUD_H, c.w, 1, rgb(60, 66, 56));

    // Local-player census.
    let mut workers = 0;
    let mut army = 0;
    for e in &w.ents {
        if e.team == w.my_team {
            match e.kind {
                Kind::Worker => workers += 1,
                k if world::is_army(k) => army += 1,
                _ => {}
            }
        }
    }
    let used = w.supply_used(w.my_team);
    let cap = w.supply_cap(w.my_team);
    let supply_col = if used + 1 >= cap {
        rgb(255, 170, 70)
    } else {
        rgb(200, 210, 220)
    };

    let minerals = w.team_min(w.my_team);
    c.text_sh(10, 8, &format!("MINERALS {}", minerals), MINERAL, 2);
    c.text_sh(200, 8, &format!("SUPPLY {}/{}", used, cap), supply_col, 2);
    c.text_sh(380, 8, &format!("WORKERS {}", workers), rgb(200, 210, 220), 2);
    c.text_sh(560, 8, &format!("ARMY {}", army), rgb(200, 210, 220), 2);
    c.text_sh(700, 8, &format!("KILLS {}", w.kills), rgb(200, 210, 220), 2);

    let mins = (w.time as i32) / 60;
    let secs = (w.time as i32) % 60;
    // How many rival factions are still in the game.
    let rivals = (0..w.factions)
        .filter(|&f| {
            let t = Team::from_idx(f);
            t != w.my_team && w.faction_alive(t)
        })
        .count();
    let right = format!("{} RIVAL{}   {}:{:02}", rivals, if rivals == 1 { "" } else { "S" }, mins, secs);
    let rw = Canvas::text_width(&right, 2);
    c.text_sh(mm_x(c.w) - rw - 10, 8, &right, edge_of(w.my_team), 2);

    // Mode indicator.
    if let Some(k) = ui.build_mode {
        c.text_sh(
            10,
            HUD_H + 8,
            &format!("PLACE {} ({})  -  ESC CANCELS", kind_name(k), cost(k)),
            rgb(120, 230, 130),
            2,
        );
    } else if ui.attack_pending {
        c.text_sh(10, HUD_H + 8, "ATTACK-MOVE: PICK A POINT", rgb(255, 200, 90), 2);
    }

    // Floating messages, stacked just under the HUD.
    for (i, (m, life)) in w.messages.iter().rev().take(4).enumerate() {
        let a = (life / 3.2).clamp(0.0, 1.0);
        let col = gfx::mix(rgb(22, 25, 20), rgb(255, 240, 200), a);
        c.text_center(c.w / 2, HUD_H + 10 + i as i32 * 18, m, col, 2);
    }

    // Selection readout + contextual hint.
    let (sw, sa, sbase, sbarr, sfact, sdepot) = w.selected_kinds();
    let mut line = String::new();
    if sbase {
        line.push_str("COMMAND CENTER: [W] WORKER  ");
    }
    if sbarr {
        line.push_str("BARRACKS: [E] SOLDIER [Y] PYRO [G] SAPPER  ");
    }
    if sfact {
        line.push_str("FACTORY: [T] TANK [R] RAIDER [V] MORTAR  ");
    }
    if sdepot {
        line.push_str("SUPPLY DEPOT: RAISES SUPPLY CAP  ");
    }
    if sbase || sbarr || sfact {
        line.push_str("(RIGHT-CLICK BLDG CANCELS QUEUE)  ");
    }
    if sw > 0 {
        line.push_str(&format!("{} WORKER: [B]ARRACKS [F]ACTORY [D]EPOT [C]CENTER  ", sw));
    }
    if sa > 0 {
        line.push_str(&format!("{} ARMY: [A] ATTACK-MOVE  [S] STOP  ", sa));
    }
    if line.is_empty() {
        line.push_str("DRAG TO SELECT  -  RIGHT-CLICK COMMAND  -  [H] HELP  -  [ESC] MENU");
    }
    c.fill_rect_a(0, c.h - 22, c.w, 22, INK, 0.8);
    c.text_sh(10, c.h - 18, &line, rgb(205, 210, 220), 2);
}

fn draw_help(c: &mut Canvas) {
    let lines = [
        "LEFT-DRAG  SELECT   DBL-CLICK  ALL OF TYPE / BLDG CLUSTER",
        "RIGHT-CLICK    MOVE / ATTACK / GATHER     A: ATTACK-MOVE",
        "MIDDLE-DRAG    PAN MAP    ARROWS / EDGES ALSO PAN",
        "CTRL+1..9      SET GROUP   1..9 RECALL (DBL = CENTER)",
        "CTRL+A         SELECT WHOLE ARMY (NO WORKERS)",
        "A   S          ATTACK-MOVE   STOP",
        "ESC            PAUSE MENU (RESUME / SURRENDER / QUIT)",
        "",
        "WORKER:  [B]ARRACKS [F]ACTORY [D]EPOT [C]COMMAND CTR",
        "  (SHIFT-CLICK WHILE PLACING TO CHAIN SEVERAL BUILDS)",
        "COMMAND CENTER: [W] WORKER      DEPOT: RAISES SUPPLY",
        "BARRACKS: [E] SOLDIER [Y] PYRO [G] SAPPER",
        "FACTORY:  [T] TANK [R] RAIDER [V] MORTAR",
        "RIGHT-CLICK A SELECTED BUILDING: CANCEL + REFUND A UNIT",
        "",
        "SOLDIER: CHEAP RANGED.    TANK: SLOW, TOUGH, SPLASH.",
        "PYRO: SHORT-RANGE FLAME CONE, MELTS CLUMPED UNITS.",
        "RAIDER: FAST, FRAGILE, HUNTS WORKERS AND FLANKS.",
        "MORTAR: LONG SIEGE SPLASH, BUT A DEAD ZONE UP CLOSE.",
        "SAPPER: SUICIDE BOMBER - CHARGES IN, BIG BLAST.",
        "HIGH GROUND: MORE SIGHT + RANGE. CLIFFS BLOCK THE",
        "VIEW FROM BELOW - HOLD A RAMP TO AMBUSH. EXPAND AND",
        "WATCH YOUR SUPPLY.   GOAL: RAZE EVERY ENEMY BASE",
    ];
    // Size the panel to its contents so nothing overlaps or overflows.
    let pad = 24;
    let top = 48;
    let line_h = 19;
    let n = lines.len() as i32;
    let maxw = lines.iter().map(|l| Canvas::text_width(l, 2)).max().unwrap_or(0);
    let bw = (maxw + pad * 2).max(520);
    let bh = top + n * line_h + 30;
    let x = (c.w - bw) / 2;
    let y = (c.h - bh) / 2;
    c.fill_rect_a(x, y, bw, bh, INK, 0.93);
    c.rect(x, y, bw, bh, rgb(90, 100, 84));
    c.text_center(c.w / 2, y + 12, "NANORTS - CONTROLS", rgb(255, 240, 200), 3);
    for (i, l) in lines.iter().enumerate() {
        c.text(x + pad, y + top + i as i32 * line_h, l, rgb(200, 208, 218), 2);
    }
    c.text_center(c.w / 2, y + top + n * line_h + 6, "PRESS H TO CLOSE", rgb(150, 160, 150), 2);
}

fn draw_gameover(c: &mut Canvas, w: &World, mouse: V2) {
    c.fill_rect_a(0, 0, c.w, c.h, INK, 0.7);
    let (title, col) = if w.over == 1 {
        ("VICTORY", rgb(110, 240, 130))
    } else {
        ("DEFEAT", rgb(240, 90, 80))
    };
    c.text_center(c.w / 2, c.h / 2 - 120, title, col, 9);
    let mins = (w.time as i32) / 60;
    let secs = (w.time as i32) % 60;
    let sub = format!("KILLS {}   TIME {}:{:02}", w.kills, mins, secs);
    c.text_center(c.w / 2, c.h / 2 - 28, &sub, rgb(220, 220, 230), 3);
    for (b, _) in gameover_buttons(c, w.versus) {
        let hover = b.hit(mouse);
        b.draw(c, hover);
    }
}
