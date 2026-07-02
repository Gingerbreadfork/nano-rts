//! Procedural audio — no sound files, everything synthesised live. Each game
//! event spawns a short-lived "voice" (an oscillator + noise + envelope) whose
//! parameters are jittered per trigger, so no two shots or explosions sound
//! exactly the same. Voices are summed in the cpal audio callback on its own
//! thread; the game just pushes `Sfx` tags into a shared queue.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// The catalogue of sounds. The game records these; the synth realises them.
#[derive(Clone, Copy)]
pub enum Sfx {
    Shot,      // soldier weapon
    TankShot,  // tank cannon
    Flame,     // flamethrower roar
    Explosion, // unit death
    BigBoom,   // building death
    Build,     // construction complete
    Train,     // unit ready
    Alarm,     // under attack
    Select,    // ui click
}

struct Voice {
    t: f32,
    dur: f32,
    phase: f32,
    freq: f32,
    freq_end: f32,
    wave: u8, // 0 sine, 1 square, 2 saw, 3 triangle
    noise: f32,
    amp0: f32,
    decay: f32,
    atk: f32,
    sr: f32,
    rng: u32,
    lg: f32, // left  channel gain (stereo pan)
    rg: f32, // right channel gain
}

#[inline]
fn rnd(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

impl Voice {
    /// `gain` (0..1) attenuates by distance from the view; `pan` (-1 left .. 1
    /// right) places it across the stereo field.
    fn spawn(sfx: Sfx, sr: f32, seed: &mut u32, gain: f32, pan: f32) -> Voice {
        let j = rnd(seed); // 0..1 jitter
        let (freq, freq_end, wave, noise, amp0, decay, atk, dur) = match sfx {
            // pitch-down zap
            Sfx::Shot => {
                let p = 680.0 + j * 280.0;
                (p, p * 0.45, 1, 0.06, 0.16, 40.0, 0.003, 0.10)
            }
            // deep cannon thump
            Sfx::TankShot => {
                let p = 150.0 + j * 50.0;
                (p, 52.0, 2, 0.5, 0.34, 13.0, 0.002, 0.24)
            }
            // breathy roar of fire — almost all noise, low and soft, fast-cycling
            // bursts overlap into a continuous whoosh while a Pyro fires
            Sfx::Flame => {
                let p = 240.0 + j * 90.0;
                (p, 150.0, 0, 0.92, 0.14, 7.0, 0.012, 0.2)
            }
            // noisy burst with a falling body
            Sfx::Explosion => {
                let p = 210.0 + j * 90.0;
                (p, 55.0, 0, 0.72, 0.36, 8.5, 0.001, 0.42)
            }
            // big low boom
            Sfx::BigBoom => {
                let p = 110.0 + j * 40.0;
                (p, 34.0, 0, 0.6, 0.52, 4.2, 0.001, 0.85)
            }
            // rising chime
            Sfx::Build => (430.0, 680.0, 3, 0.0, 0.13, 7.0, 0.005, 0.32),
            // blip up
            Sfx::Train => {
                let p = 560.0 + j * 160.0;
                (p, p * 1.35, 1, 0.0, 0.12, 20.0, 0.003, 0.09)
            }
            // steady warning square
            Sfx::Alarm => (360.0, 300.0, 1, 0.0, 0.17, 2.2, 0.01, 0.45),
            // tiny click
            Sfx::Select => (520.0, 520.0, 0, 0.0, 0.08, 45.0, 0.002, 0.03),
        };
        let pan = pan.clamp(-1.0, 1.0);
        Voice {
            t: 0.0,
            dur,
            phase: 0.0,
            freq,
            freq_end,
            wave,
            noise,
            amp0: amp0 * gain.clamp(0.0, 1.0), // distance attenuation
            decay,
            atk,
            sr,
            rng: *seed | 1,
            // Keep the near side at full volume; fade only the far side.
            lg: 1.0 - pan.max(0.0) * 0.6,
            rg: 1.0 + pan.min(0.0) * 0.6,
        }
    }

    #[inline]
    fn noise_sample(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    #[inline]
    fn next(&mut self) -> f32 {
        let dt = 1.0 / self.sr;
        let frac = (self.t / self.dur).min(1.0);
        let f = self.freq + (self.freq_end - self.freq) * frac;
        self.phase = (self.phase + f * dt).fract();
        let osc = match self.wave {
            1 => {
                if self.phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            2 => 2.0 * self.phase - 1.0,
            3 => 1.0 - 4.0 * (self.phase - 0.5).abs(),
            _ => (self.phase * std::f32::consts::TAU).sin(),
        };
        let n = self.noise_sample();
        let body = osc * (1.0 - self.noise) + n * self.noise;
        let env = (self.t / self.atk).min(1.0) * (-self.decay * self.t).exp();
        // Slow-decay voices (Alarm ends near 37% of peak) would otherwise be
        // cut mid-swing at t == dur — an audible click. A ~3 ms linear release
        // ramps every voice to exactly zero at its end.
        let rel = ((self.dur - self.t) * (1.0 / 0.003)).clamp(0.0, 1.0);
        self.t += dt;
        body * self.amp0 * env * rel
    }

    #[inline]
    fn done(&self) -> bool {
        self.t >= self.dur
    }
}

/// A queued sound: the tag, plus its distance gain and stereo pan.
type Cue = (Sfx, f32, f32);

pub struct Audio {
    queue: Arc<Mutex<VecDeque<Cue>>>,
    _stream: cpal::Stream,
}

impl Audio {
    /// Open the default output and start the synth. Returns None (silent) if no
    /// audio device is available — the game runs fine either way.
    pub fn new() -> Option<Audio> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let supported = device.default_output_config().ok()?;
        let sr = supported.sample_rate().0 as f32;
        let channels = supported.channels() as usize;
        let config = supported.config();
        let queue: Arc<Mutex<VecDeque<Cue>>> = Arc::new(Mutex::new(VecDeque::new()));

        // Cover every format the generic synth can render into — some devices
        // default to I32 or F64, and matching only three would leave them mute.
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => build::<f32>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::F64 => build::<f64>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::I8 => build::<i8>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::I16 => build::<i16>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::I32 => build::<i32>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::I64 => build::<i64>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::U8 => build::<u8>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::U16 => build::<u16>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::U32 => build::<u32>(&device, &config, sr, channels, queue.clone()),
            cpal::SampleFormat::U64 => build::<u64>(&device, &config, sr, channels, queue.clone()),
            _ => None,
        }?;
        stream.play().ok()?;
        Some(Audio { queue, _stream: stream })
    }

    /// Play `sfx` at `gain` (0..1 distance attenuation) and `pan` (-1..1).
    /// Fully-attenuated sounds are dropped so they don't crowd the voice pool.
    pub fn play(&self, sfx: Sfx, gain: f32, pan: f32) {
        if gain <= 0.001 {
            return;
        }
        if let Ok(mut q) = self.queue.lock() {
            if q.len() < 96 {
                q.push_back((sfx, gain, pan));
            }
        }
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sr: f32,
    channels: usize,
    queue: Arc<Mutex<VecDeque<Cue>>>,
) -> Option<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    // How many voices may sound at once; the oldest is evicted past this.
    const MAX_VOICES: usize = 48;
    // The callback runs on the real-time audio thread, so the pool is sized
    // up front — pushing must never reallocate (malloc there risks underruns).
    let mut voices: Vec<Voice> = Vec::with_capacity(MAX_VOICES);
    let mut seed: u32 = 0x9E37_79B9;
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                // Never block the audio thread on the game thread: if the queue
                // is contended right now, skip draining — cues stay put and the
                // next callback picks them up, while live voices keep playing.
                if let Ok(mut q) = queue.try_lock() {
                    while let Some((s, gain, pan)) = q.pop_front() {
                        if voices.len() >= MAX_VOICES {
                            voices.remove(0);
                        }
                        voices.push(Voice::spawn(s, sr, &mut seed, gain, pan));
                    }
                }
                let stereo = channels >= 2;
                for frame in data.chunks_mut(channels.max(1)) {
                    let (mut sl, mut sr2) = (0.0f32, 0.0f32);
                    for v in voices.iter_mut() {
                        let x = v.next();
                        sl += x * v.lg;
                        sr2 += x * v.rg;
                    }
                    sl = (sl * 0.6).clamp(-1.0, 1.0);
                    sr2 = (sr2 * 0.6).clamp(-1.0, 1.0);
                    if stereo {
                        frame[0] = T::from_sample(sl);
                        frame[1] = T::from_sample(sr2);
                        let mid = T::from_sample(((sl + sr2) * 0.5).clamp(-1.0, 1.0));
                        for c in frame.iter_mut().skip(2) {
                            *c = mid;
                        }
                    } else {
                        let mono = T::from_sample(((sl + sr2) * 0.5).clamp(-1.0, 1.0));
                        for c in frame.iter_mut() {
                            *c = mono;
                        }
                    }
                }
                voices.retain(|v| !v.done());
            },
            |e| eprintln!("audio stream error: {e}"),
            None,
        )
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_voice_releases_to_silence() {
        // A voice cut while its envelope is still large is a hard step in the
        // output — an audible click. The release ramp must bring the very last
        // sample of every sound essentially to zero, Alarm especially (its
        // slow decay leaves it near 37% of peak at t == dur).
        let sr = 48_000.0;
        for sfx in [
            Sfx::Shot,
            Sfx::TankShot,
            Sfx::Flame,
            Sfx::Explosion,
            Sfx::BigBoom,
            Sfx::Build,
            Sfx::Train,
            Sfx::Alarm,
            Sfx::Select,
        ] {
            let mut seed = 0x1234_5678u32;
            let mut v = Voice::spawn(sfx, sr, &mut seed, 1.0, 0.0);
            let mut peak = 0.0f32;
            let mut last = 0.0f32;
            while !v.done() {
                last = v.next();
                peak = peak.max(last.abs());
            }
            assert!(peak > 0.01, "voice should be audible at its peak");
            assert!(
                last.abs() < 0.005,
                "voice must end at silence, got {last} (peak {peak})"
            );
        }
    }
}
