//! The entire math library. Two floats and the handful of operations a tiny
//! RTS actually needs.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V2 {
    pub x: f32,
    pub y: f32,
}

#[inline]
pub fn v2(x: f32, y: f32) -> V2 {
    V2 { x, y }
}

impl V2 {
    #[inline]
    pub fn add(self, o: V2) -> V2 {
        v2(self.x + o.x, self.y + o.y)
    }
    #[inline]
    pub fn sub(self, o: V2) -> V2 {
        v2(self.x - o.x, self.y - o.y)
    }
    #[inline]
    pub fn scale(self, s: f32) -> V2 {
        v2(self.x * s, self.y * s)
    }
    #[inline]
    pub fn len_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
    #[inline]
    pub fn len(self) -> f32 {
        self.len_sq().sqrt()
    }
    /// Unit vector, or zero for a zero-length input (no NaNs to chase).
    #[inline]
    pub fn norm(self) -> V2 {
        let l = self.len();
        if l > 1e-6 {
            v2(self.x / l, self.y / l)
        } else {
            v2(0.0, 0.0)
        }
    }
    #[inline]
    pub fn dist_sq(self, o: V2) -> f32 {
        self.sub(o).len_sq()
    }
    #[inline]
    pub fn dist(self, o: V2) -> f32 {
        self.sub(o).len()
    }
}

// ---- deterministic trig -----------------------------------------------------
// std's f32 sin/cos call the platform libm, which is NOT bit-identical across
// OSes — poison for anything that feeds the lockstep checksum. These evaluate a
// fixed-coefficient polynomial using only f32 mul/add/sub and floor (all
// IEEE-exact), so every peer computes the same bits. Accuracy is ~1e-4 over a
// few dozen radians — ample for layout, useless for astronomy.

const TAU: f32 = 6.2831855;
const INV_TAU: f32 = 0.15915494;

pub fn det_sin(x: f32) -> f32 {
    // Range-reduce to turns in [-0.5, 0.5), then fold onto [-0.25, 0.25]
    // (sine's symmetric quarter-wave), and evaluate an odd polynomial there.
    let mut t = x * INV_TAU;
    t -= (t + 0.5).floor();
    if t > 0.25 {
        t = 0.5 - t;
    }
    if t < -0.25 {
        t = -0.5 - t;
    }
    let y = t * TAU; // now in [-pi/2, pi/2]
    let y2 = y * y;
    y * (1.0 + y2 * (-0.16666667 + y2 * (0.008333333 + y2 * -0.00019841270)))
}

pub fn det_cos(x: f32) -> f32 {
    det_sin(x + 1.5707964)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_trig_tracks_std_within_tolerance() {
        // A dense sweep across many periods, including awkward fold points.
        let mut x = -30.0f32;
        while x < 30.0 {
            let (s, c) = (det_sin(x), det_cos(x));
            assert!((s - x.sin()).abs() < 1e-3, "det_sin({x}) = {s} vs {}", x.sin());
            assert!((c - x.cos()).abs() < 1e-3, "det_cos({x}) = {c} vs {}", x.cos());
            x += 0.0137;
        }
    }
}
