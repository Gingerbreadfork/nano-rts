//! A from-scratch software renderer. We own a flat `u32` ARGB buffer and draw
//! everything ourselves: rects, circles, diamonds, lines, and text. No GPU, no
//! sprites on disk — the whole look of the game lives in this file.

use crate::font;
use crate::vec::V2;

pub struct Canvas {
    pub w: i32,
    pub h: i32,
    pub buf: Vec<u32>,
}

#[inline]
pub fn rgb(r: u8, g: u8, b: u8) -> u32 {
    0xFF00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Linear blend between two packed colors. `t` in 0..=1 picks `b`.
#[inline]
pub fn mix(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let ar = ((a >> 16) & 0xFF) as f32;
    let ag = ((a >> 8) & 0xFF) as f32;
    let ab = (a & 0xFF) as f32;
    let br = ((b >> 16) & 0xFF) as f32;
    let bg = ((b >> 8) & 0xFF) as f32;
    let bb = (b & 0xFF) as f32;
    rgb(
        (ar + (br - ar) * t) as u8,
        (ag + (bg - ag) * t) as u8,
        (ab + (bb - ab) * t) as u8,
    )
}

impl Canvas {
    pub fn new(w: i32, h: i32) -> Canvas {
        Canvas {
            w,
            h,
            buf: vec![0xFF00_0000; (w * h) as usize],
        }
    }

    #[inline]
    pub fn clear(&mut self, color: u32) {
        for p in self.buf.iter_mut() {
            *p = color;
        }
    }

    #[inline]
    pub fn put(&mut self, x: i32, y: i32, color: u32) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h {
            self.buf[(y * self.w + x) as usize] = color;
        }
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.w);
        let y1 = (y + h).min(self.h);
        for yy in y0..y1 {
            let row = yy * self.w;
            for xx in x0..x1 {
                self.buf[(row + xx) as usize] = color;
            }
        }
    }

    /// Alpha-blended fill. This runs over full-screen spans every frame (fog
    /// cells, damage flash, pause dim), so it blends in fixed point: alpha is
    /// quantised to 0..=256 once, the rect is clamped once, and the inner loop
    /// is three integer lerps per pixel with no bounds checks. May differ from
    /// the float `mix` path by at most 1/255 per channel.
    pub fn fill_rect_a(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32, a: f32) {
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.w);
        let y1 = (y + h).min(self.h);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        // 256 (not 255) so a == 1.0 replaces the background exactly via >> 8.
        let ai = (a.clamp(0.0, 1.0) * 256.0 + 0.5) as u32;
        if ai == 0 {
            return;
        }
        let inv = 256 - ai;
        let fr = ((color >> 16) & 0xFF) * ai;
        let fg = ((color >> 8) & 0xFF) * ai;
        let fb = (color & 0xFF) * ai;
        for yy in y0..y1 {
            let row = (yy * self.w) as usize;
            for px in self.buf[row + x0 as usize..row + x1 as usize].iter_mut() {
                let bg = *px;
                let r = (((bg >> 16) & 0xFF) * inv + fr) >> 8;
                let g = (((bg >> 8) & 0xFF) * inv + fg) >> 8;
                let b = ((bg & 0xFF) * inv + fb) >> 8;
                *px = 0xFF00_0000 | (r << 16) | (g << 8) | b;
            }
        }
    }

    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        self.fill_rect(x, y, w, 1, color);
        self.fill_rect(x, y + h - 1, w, 1, color);
        self.fill_rect(x, y, 1, h, color);
        self.fill_rect(x + w - 1, y, 1, h, color);
    }

    pub fn fill_circle(&mut self, cx: i32, cy: i32, r: i32, color: u32) {
        if r <= 0 {
            self.put(cx, cy, color);
            return;
        }
        let r2 = r * r;
        for dy in -r..=r {
            let yy = cy + dy;
            if yy < 0 || yy >= self.h {
                continue;
            }
            let span = ((r2 - dy * dy) as f32).sqrt() as i32;
            let x0 = (cx - span).max(0);
            let x1 = (cx + span).min(self.w - 1);
            let row = yy * self.w;
            for xx in x0..=x1 {
                self.buf[(row + xx) as usize] = color;
            }
        }
    }

    /// Additive blend into an already-validated buffer index. The span
    /// fillers below clip once per row and then call this in their inner
    /// loops, skipping the per-pixel bounds check `add_px` pays.
    #[inline]
    fn add_at(&mut self, i: usize, color: u32, t: f32) {
        let bg = self.buf[i];
        let r = (((bg >> 16) & 0xFF) as f32 + ((color >> 16) & 0xFF) as f32 * t).min(255.0) as u32;
        let g = (((bg >> 8) & 0xFF) as f32 + ((color >> 8) & 0xFF) as f32 * t).min(255.0) as u32;
        let b = ((bg & 0xFF) as f32 + (color & 0xFF) as f32 * t).min(255.0) as u32;
        self.buf[i] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
    }

    /// Additive pixel: adds `color * t` to whatever's there (clamped). Lets
    /// overlapping glow particles stack and blow out to white.
    #[inline]
    pub fn add_px(&mut self, x: i32, y: i32, color: u32, t: f32) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h {
            self.add_at((y * self.w + x) as usize, color, t);
        }
    }

    /// Additive horizontal run, clamped to the canvas. `y` must already be
    /// on-screen; x0..=x1 may hang off either edge.
    #[inline]
    fn add_span(&mut self, x0: i32, x1: i32, y: i32, color: u32, t: f32) {
        let x0 = x0.max(0);
        let x1 = x1.min(self.w - 1);
        let row = y * self.w;
        for xx in x0..=x1 {
            self.add_at((row + xx) as usize, color, t);
        }
    }

    /// Additive filled circle — a glowing blob.
    pub fn fill_circle_add(&mut self, cx: i32, cy: i32, r: i32, color: u32, t: f32) {
        if r <= 0 {
            self.add_px(cx, cy, color, t);
            return;
        }
        let r2 = r * r;
        for dy in -r..=r {
            let yy = cy + dy;
            if yy < 0 || yy >= self.h {
                continue;
            }
            let span = ((r2 - dy * dy) as f32).sqrt() as i32;
            self.add_span(cx - span, cx + span, yy, color, t);
        }
    }

    /// Additive ring (annulus between `r0` and `r1`) — a shockwave. Each row
    /// is at most two horizontal runs (the ring's left and right arcs), so we
    /// walk only those instead of testing the whole bounding square.
    pub fn ring_add(&mut self, cx: i32, cy: i32, r0: i32, r1: i32, color: u32, t: f32) {
        let r1 = r1.max(r0 + 1);
        let (o0, o1) = (r0 * r0, r1 * r1);
        for dy in -r1..=r1 {
            let yy = cy + dy;
            if yy < 0 || yy >= self.h {
                continue;
            }
            // Outer edge: widest |dx| with dx² + dy² <= o1.
            let out = ((o1 - dy * dy) as f32).sqrt() as i32;
            // Inner hole: widest |dx| with dx² + dy² < o0, or none if the row
            // sits entirely at or beyond the inner radius.
            let k = o0 - dy * dy;
            let hole = if k > 0 {
                let mut s = (k as f32).sqrt() as i32;
                if s * s >= k {
                    s -= 1; // dx = s lands on the ring itself, not the hole
                }
                s
            } else {
                -1
            };
            if hole < 0 {
                self.add_span(cx - out, cx + out, yy, color, t);
            } else {
                self.add_span(cx - out, cx - hole - 1, yy, color, t);
                self.add_span(cx + hole + 1, cx + out, yy, color, t);
            }
        }
    }

    pub fn circle(&mut self, cx: i32, cy: i32, r: i32, color: u32) {
        // Midpoint circle outline.
        let mut x = r;
        let mut y = 0;
        let mut err = 1 - r;
        while x >= y {
            for &(px, py) in &[
                (cx + x, cy + y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx - x, cy + y),
                (cx - x, cy - y),
                (cx - y, cy - x),
                (cx + y, cy - x),
                (cx + x, cy - y),
            ] {
                self.put(px, py, color);
            }
            y += 1;
            if err < 0 {
                err += 2 * y + 1;
            } else {
                x -= 1;
                err += 2 * (y - x) + 1;
            }
        }
    }

    /// A filled diamond (rotated square) — our soldier silhouette.
    pub fn fill_diamond(&mut self, cx: i32, cy: i32, r: i32, color: u32) {
        for dy in -r..=r {
            let yy = cy + dy;
            if yy < 0 || yy >= self.h {
                continue;
            }
            let span = r - dy.abs();
            let x0 = (cx - span).max(0);
            let x1 = (cx + span).min(self.w - 1);
            let row = yy * self.w;
            for xx in x0..=x1 {
                self.buf[(row + xx) as usize] = color;
            }
        }
    }

    /// Filled rectangle oriented along `dir` (a unit vector): `half_len` along
    /// the forward axis, `half_wid` across it. Used for rotating tank hulls and
    /// gun barrels so units face where they're going.
    pub fn fill_orect(&mut self, cx: i32, cy: i32, half_len: f32, half_wid: f32, dir: V2, color: u32) {
        let (fx, fy) = (dir.x, dir.y);
        let r = half_len.hypot(half_wid).ceil() as i32 + 1;
        for dy in -r..=r {
            for dx in -r..=r {
                let along = dx as f32 * fx + dy as f32 * fy;
                let perp = dx as f32 * -fy + dy as f32 * fx;
                if along.abs() <= half_len && perp.abs() <= half_wid {
                    self.put(cx + dx, cy + dy, color);
                }
            }
        }
    }

    /// Filled triangle via edge functions over the bounding box.
    pub fn fill_tri(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, x2: i32, y2: i32, color: u32) {
        let minx = x0.min(x1).min(x2).max(0);
        let maxx = x0.max(x1).max(x2).min(self.w - 1);
        let miny = y0.min(y1).min(y2).max(0);
        let maxy = y0.max(y1).max(y2).min(self.h - 1);
        for y in miny..=maxy {
            for x in minx..=maxx {
                let w0 = (x1 - x) * (y2 - y) - (x2 - x) * (y1 - y);
                let w1 = (x2 - x) * (y0 - y) - (x0 - x) * (y2 - y);
                let w2 = (x0 - x) * (y1 - y) - (x1 - x) * (y0 - y);
                let neg = w0 < 0 || w1 < 0 || w2 < 0;
                let pos = w0 > 0 || w1 > 0 || w2 > 0;
                if !(neg && pos) {
                    self.put(x, y, color);
                }
            }
        }
    }

    pub fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut x = x0;
        let mut y = y0;
        loop {
            self.put(x, y, color);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Dashed line, for rally points and move orders.
    pub fn dashed_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32, dash: i32) {
        let dx = (x1 - x0) as f32;
        let dy = (y1 - y0) as f32;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1.0 {
            return;
        }
        let steps = len as i32;
        let dash = dash.max(1); // a zero dash would divide by zero below
        for i in 0..=steps {
            if (i / dash) % 2 == 0 {
                let t = i as f32 / steps as f32;
                self.put(
                    x0 + (dx * t) as i32,
                    y0 + (dy * t) as i32,
                    color,
                );
            }
        }
    }

    pub fn glyph(&mut self, x: i32, y: i32, c: char, color: u32, scale: i32) {
        let g = font::glyph(c);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..font::GLYPH_W {
                if bits & (1 << (font::GLYPH_W - 1 - col)) != 0 {
                    self.fill_rect(
                        x + col * scale,
                        y + row as i32 * scale,
                        scale,
                        scale,
                        color,
                    );
                }
            }
        }
    }

    pub fn text(&mut self, x: i32, y: i32, s: &str, color: u32, scale: i32) {
        let mut cx = x;
        for c in s.chars() {
            self.glyph(cx, y, c, color, scale);
            cx += font::ADVANCE * scale;
        }
    }

    /// Text with a 1px drop shadow for legibility over busy backgrounds.
    pub fn text_sh(&mut self, x: i32, y: i32, s: &str, color: u32, scale: i32) {
        self.text(x + scale, y + scale, s, rgb(0, 0, 0), scale);
        self.text(x, y, s, color, scale);
    }

    pub fn text_width(s: &str, scale: i32) -> i32 {
        s.chars().count() as i32 * font::ADVANCE * scale
    }

    pub fn text_center(&mut self, cx: i32, y: i32, s: &str, color: u32, scale: i32) {
        let w = Canvas::text_width(s, scale);
        self.text_sh(cx - w / 2, y, s, color, scale);
    }
}

// ---- bloom ------------------------------------------------------------------
//
// A post-process glow in integer math: downsample what clears a luma threshold,
// box-blur the small buffer, then upsample and add it back over the frame.

const BLOOM_SCALE: usize = 4; // downsample factor
// Luma below which nothing glows, on a 0..255 scale. Must stay >= 128: the
// downsample rejects dark pixels with a `p & 0x808080` test that assumes it.
const BLOOM_THRESHOLD: u32 = 168;
const BLOOM_STRENGTH: u32 = 150; // /256 gain applied to the brightpass

/// Reusable bloom pass. Its scratch buffers are rebuilt only when the canvas
/// size changes, so applying it each frame allocates nothing.
pub struct Bloom {
    cw: usize, // canvas size the buffers currently match
    ch: usize,
    w: usize, // downsampled size
    h: usize,
    a: Vec<u32>, // brightpass and blur ping-pong buffers, packed 00RRGGBB
    b: Vec<u32>,
    xmap: Vec<(usize, usize, u32)>, // per canvas column: (i0, i1, fx 0..=256)
    row0: Vec<u32>, // bilinear-expanded small rows cached for the composite
    row1: Vec<u32>,
}

/// Bilinear helper: per-channel `(c0*(256-f) + c1*f) >> 8` on packed pixels.
#[inline]
fn lerp_px(p0: u32, p1: u32, f: u32) -> u32 {
    let inv = 256 - f;
    let r = (((p0 >> 16) & 0xFF) * inv + ((p1 >> 16) & 0xFF) * f) >> 8;
    let g = (((p0 >> 8) & 0xFF) * inv + ((p1 >> 8) & 0xFF) * f) >> 8;
    let b = ((p0 & 0xFF) * inv + (p1 & 0xFF) * f) >> 8;
    (r << 16) | (g << 8) | b
}

/// One 5-tap box-blur pass over the small buffer, edge-replicated. Keeps a
/// sliding window sum, one pixel in and one out, instead of five taps each.
fn box_pass(src: &[u32], dst: &mut [u32], w: usize, h: usize, horiz: bool) {
    let (len, n, stride) = if horiz { (w, h, 1) } else { (h, w, w) };
    let li = len as i32;
    for lane in 0..n {
        let base = if horiz { lane * w } else { lane };
        let at = |i: i32| -> u32 { src[base + i.clamp(0, li - 1) as usize * stride] };
        let (mut r, mut g, mut b) = (0i32, 0i32, 0i32);
        for i in -2..=2 {
            let p = at(i);
            r += ((p >> 16) & 0xFF) as i32;
            g += ((p >> 8) & 0xFF) as i32;
            b += (p & 0xFF) as i32;
        }
        for i in 0..len {
            dst[base + i * stride] = ((r as u32 / 5) << 16) | ((g as u32 / 5) << 8) | (b as u32 / 5);
            let out = at(i as i32 - 2);
            let inp = at(i as i32 + 3);
            r += ((inp >> 16) & 0xFF) as i32 - ((out >> 16) & 0xFF) as i32;
            g += ((inp >> 8) & 0xFF) as i32 - ((out >> 8) & 0xFF) as i32;
            b += (inp & 0xFF) as i32 - (out & 0xFF) as i32;
        }
    }
}

impl Bloom {
    pub fn new() -> Bloom {
        Bloom { cw: 0, ch: 0, w: 0, h: 0, a: Vec::new(), b: Vec::new(), xmap: Vec::new(), row0: Vec::new(), row1: Vec::new() }
    }

    /// Rebuild the scratch buffers when the canvas size differs from `cw` x `ch`.
    fn ensure(&mut self, cw: usize, ch: usize) {
        if self.cw == cw && self.ch == ch {
            return;
        }
        self.cw = cw;
        self.ch = ch;
        self.w = cw.div_ceil(BLOOM_SCALE).max(2);
        self.h = ch.div_ceil(BLOOM_SCALE).max(2);
        self.a = vec![0; self.w * self.h];
        self.b = vec![0; self.w * self.h];
        self.row0 = vec![0; cw];
        self.row1 = vec![0; cw];
        // Upsample x-mapping: which two small columns each canvas column
        // blends, and the 8.8 fixed-point weight between them.
        self.xmap = (0..cw)
            .map(|x| {
                let pos = (x as i32 * 256 + 128) / BLOOM_SCALE as i32 - 128;
                let pos = pos.max(0) as usize;
                let i0 = (pos >> 8).min(self.w - 1);
                (i0, (i0 + 1).min(self.w - 1), (pos & 255) as u32)
            })
            .collect();
    }

    /// Expand small row `r` to canvas width with horizontal bilinear.
    /// Returns whether anything in the row glows at all.
    fn expand(a: &[u32], xmap: &[(usize, usize, u32)], w: usize, r: usize, dst: &mut [u32]) -> bool {
        let base = r * w;
        let mut any = 0u32;
        for (d, &(i0, i1, f)) in dst.iter_mut().zip(xmap.iter()) {
            *d = lerp_px(a[base + i0], a[base + i1], f);
            any |= *d;
        }
        any != 0
    }

    /// Apply the glow to a frame in place.
    pub fn apply(&mut self, c: &mut Canvas) {
        let (cw, ch) = (c.w as usize, c.h as usize);
        if cw < BLOOM_SCALE * 2 || ch < BLOOM_SCALE * 2 {
            return;
        }
        self.ensure(cw, ch);
        let (w, h) = (self.w, self.h);

        // 1. Downsample and threshold. Each pixel is thresholded before the
        // block is averaged, so a single hot pixel still deposits its light.
        for gy in 0..h {
            let y0 = gy * BLOOM_SCALE;
            let y1 = (y0 + BLOOM_SCALE).min(ch);
            for gx in 0..w {
                let x0 = gx * BLOOM_SCALE;
                let x1 = (x0 + BLOOM_SCALE).min(cw);
                let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
                for yy in y0..y1 {
                    let row = yy * cw;
                    for &p in &c.buf[row + x0..row + x1] {
                        if p & 0x0080_8080 == 0 {
                            continue; // every channel < 128, so luma < 128 <= threshold
                        }
                        let (pr, pg, pb) = ((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF);
                        let luma = (pr * 54 + pg * 183 + pb * 19) >> 8;
                        if luma > BLOOM_THRESHOLD {
                            let k = (luma - BLOOM_THRESHOLD) * 256 / (255 - BLOOM_THRESHOLD); // 0..=256 knee
                            r += (pr * k) >> 8;
                            g += (pg * k) >> 8;
                            b += (pb * k) >> 8;
                        }
                    }
                }
                let n = ((x1 - x0) * (y1 - y0)) as u32;
                // Strength may exceed unity, so clamp each channel to a byte.
                let r = (r * BLOOM_STRENGTH / (n << 8)).min(255);
                let g = (g * BLOOM_STRENGTH / (n << 8)).min(255);
                let b = (b * BLOOM_STRENGTH / (n << 8)).min(255);
                self.a[gy * w + gx] = (r << 16) | (g << 8) | b;
            }
        }

        // 2. Two box-blur rounds, horizontal then vertical, for a soft halo.
        for _ in 0..2 {
            box_pass(&self.a, &mut self.b, w, h, true);
            box_pass(&self.b, &mut self.a, w, h, false);
        }

        // 3. Bilinear upsample, added over the frame with saturation. The two
        // small rows in play are expanded once and reused until they change.
        let mut cached: i32 = -1;
        let (mut lit0, mut lit1) = (false, false);
        for y in 0..ch {
            let pos = (y as i32 * 256 + 128) / BLOOM_SCALE as i32 - 128;
            let pos = pos.max(0) as usize;
            let r0 = (pos >> 8).min(h - 1);
            let fy = (pos & 255) as u32;
            let r1 = (r0 + 1).min(h - 1);
            if r0 as i32 != cached {
                lit0 = Bloom::expand(&self.a, &self.xmap, w, r0, &mut self.row0);
                lit1 = Bloom::expand(&self.a, &self.xmap, w, r1, &mut self.row1);
                cached = r0 as i32;
            }
            if !lit0 && !lit1 {
                continue; // this whole stripe of the frame has no glow
            }
            let row = y * cw;
            for x in 0..cw {
                let (p0, p1) = (self.row0[x], self.row1[x]);
                if p0 | p1 == 0 {
                    continue; // nothing glowing here
                }
                let glow = lerp_px(p0, p1, fy);
                let bg = c.buf[row + x];
                let r = (((bg >> 16) & 0xFF) + ((glow >> 16) & 0xFF)).min(255);
                let g = (((bg >> 8) & 0xFF) + ((glow >> 8) & 0xFF)).min(255);
                let b = ((bg & 0xFF) + (glow & 0xFF)).min(255);
                c.buf[row + x] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
            }
        }
    }
}

impl Default for Bloom {
    fn default() -> Bloom {
        Bloom::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashed_line_tolerates_zero_dash() {
        let mut c = Canvas::new(32, 32);
        // A zero dash length must not divide by zero; it degrades to solid.
        c.dashed_line(0, 0, 20, 10, rgb(255, 0, 0), 0);
        assert_eq!(c.buf[0], rgb(255, 0, 0));
    }

    /// Brute-force additive ring, pixel-tested over the whole bounding square
    /// — the shape the span version must reproduce exactly.
    fn ring_add_ref(c: &mut Canvas, cx: i32, cy: i32, r0: i32, r1: i32, color: u32, t: f32) {
        let r1 = r1.max(r0 + 1);
        let (o0, o1) = (r0 * r0, r1 * r1);
        for dy in -r1..=r1 {
            for dx in -r1..=r1 {
                let d2 = dx * dx + dy * dy;
                if d2 >= o0 && d2 <= o1 {
                    c.add_px(cx + dx, cy + dy, color, t);
                }
            }
        }
    }

    #[test]
    fn ring_add_matches_per_pixel_reference() {
        // Include rings hanging off every edge and degenerate radii.
        for &(cx, cy, r0, r1) in &[
            (16, 16, 4, 7),
            (16, 16, 0, 5),
            (16, 16, 5, 5), // r1 clamps up to r0 + 1
            (-2, 3, 3, 8),
            (31, 31, 2, 9),
            (16, -5, 4, 12),
        ] {
            let mut a = Canvas::new(32, 32);
            let mut b = Canvas::new(32, 32);
            a.ring_add(cx, cy, r0, r1, rgb(200, 120, 40), 0.7);
            ring_add_ref(&mut b, cx, cy, r0, r1, rgb(200, 120, 40), 0.7);
            assert_eq!(a.buf, b.buf, "ring ({cx},{cy}) r{r0}..r{r1} diverged");
        }
    }

    #[test]
    fn fill_circle_add_clips_offscreen_spans() {
        // Rows overhang both horizontal edges; must match the checked path.
        for &(cx, cy) in &[(1, 10), (30, 10), (-3, 5), (34, 20)] {
            let mut a = Canvas::new(32, 32);
            let mut b = Canvas::new(32, 32);
            a.fill_circle_add(cx, cy, 6, rgb(90, 200, 250), 0.5);
            let r2 = 36;
            for dy in -6..=6i32 {
                let yy = cy + dy;
                let span = ((r2 - dy * dy) as f32).sqrt() as i32;
                for xx in (cx - span)..=(cx + span) {
                    b.add_px(xx, yy, rgb(90, 200, 250), 0.5);
                }
            }
            assert_eq!(a.buf, b.buf, "circle at ({cx},{cy}) diverged");
        }
    }

    /// The sliding-window blur must reproduce the naive edge-clamped 5-tap
    /// box filter exactly, both directions, including the edges.
    #[test]
    fn box_pass_matches_naive_reference() {
        let (w, h) = (13usize, 7usize);
        let src: Vec<u32> = (0..w * h)
            .map(|i| {
                let x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                (x >> 24) as u32 & 0x00FF_FFFF
            })
            .collect();
        for horiz in [true, false] {
            let mut fast = vec![0; w * h];
            box_pass(&src, &mut fast, w, h, horiz);
            let mut naive = vec![0; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
                    for d in -2i32..=2 {
                        let (sx, sy) = if horiz {
                            ((x as i32 + d).clamp(0, w as i32 - 1) as usize, y)
                        } else {
                            (x, (y as i32 + d).clamp(0, h as i32 - 1) as usize)
                        };
                        let p = src[sy * w + sx];
                        r += (p >> 16) & 0xFF;
                        g += (p >> 8) & 0xFF;
                        b += p & 0xFF;
                    }
                    naive[y * w + x] = ((r / 5) << 16) | ((g / 5) << 8) | (b / 5);
                }
            }
            assert_eq!(fast, naive, "horiz={horiz}");
        }
    }

    /// A bright block must bleed light onto nearby dark pixels, leave far
    /// pixels untouched, and never darken the source.
    #[test]
    fn bloom_spreads_bright_light() {
        let mut c = Canvas::new(96, 96);
        c.clear(rgb(10, 10, 10));
        c.fill_rect(44, 44, 8, 8, rgb(255, 255, 255));
        let mut bl = Bloom::new();
        bl.apply(&mut c);
        // Just outside the block: glow landed.
        let near = c.buf[48 * 96 + 58];
        assert!(near & 0xFF > 10, "near pixel should be lit by glow, got {near:08x}");
        // A far corner: unchanged.
        assert_eq!(c.buf[5 * 96 + 5], rgb(10, 10, 10), "far pixel must not change");
        // The source block itself stays fully bright.
        assert_eq!(c.buf[48 * 96 + 48], rgb(255, 255, 255));
    }

    /// A frame with nothing over the threshold passes through untouched.
    #[test]
    fn bloom_ignores_dim_scenes() {
        let mut c = Canvas::new(64, 48);
        c.clear(rgb(60, 70, 80));
        c.fill_rect(10, 10, 20, 20, rgb(120, 90, 100));
        let before = c.buf.clone();
        let mut bl = Bloom::new();
        bl.apply(&mut c);
        assert_eq!(c.buf, before, "below-threshold content must not glow");
    }

    /// Sizes that are not multiples of the downsample factor, and reuse
    /// across a resize, must not panic or index out of bounds.
    #[test]
    fn bloom_survives_odd_sizes_and_resizes() {
        let mut bl = Bloom::new();
        for &(w, h) in &[(33, 17), (129, 71), (640, 360), (7, 7), (1281, 721)] {
            let mut c = Canvas::new(w, h);
            c.clear(rgb(0, 0, 0));
            c.fill_rect(w / 3, h / 3, 6, 6, rgb(255, 240, 200));
            bl.apply(&mut c);
        }
    }

    #[test]
    fn fill_rect_a_tracks_float_mix_within_one_step() {
        let bg = rgb(37, 91, 143);
        let fg = rgb(240, 180, 20);
        for &a in &[0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let mut c = Canvas::new(8, 8);
            c.clear(bg);
            // Deliberately overhang all four edges to exercise the clamp.
            c.fill_rect_a(-4, -4, 16, 16, fg, a);
            let want = mix(bg, fg, a);
            let got = c.buf[0];
            for shift in [16u32, 8, 0] {
                let w = (want >> shift) & 0xFF;
                let g = (got >> shift) & 0xFF;
                assert!(
                    (w as i32 - g as i32).abs() <= 1,
                    "alpha {a}: channel {shift} off by more than 1 ({w} vs {g})"
                );
            }
        }
        // The endpoints must be exact: 0 leaves the buffer alone, 1 replaces.
        let mut c = Canvas::new(4, 4);
        c.clear(bg);
        c.fill_rect_a(0, 0, 4, 4, fg, 0.0);
        assert_eq!(c.buf[5], bg);
        c.fill_rect_a(0, 0, 4, 4, fg, 1.0);
        assert_eq!(c.buf[5], fg);
    }
}
