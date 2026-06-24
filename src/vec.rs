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
