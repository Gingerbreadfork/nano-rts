//! Replay recording and playback. The sim is deterministic and every player
//! action is a `world::Cmd`, so a match is its seed and config plus each
//! command stamped with the sim step it landed on. Command records reuse the
//! wire codec byte for byte.
//!
//! File layout of `replays/nanorts-YYYYMMDD-HHMMSS-<mode>.nrep`, little-endian:
//!   "NREP" u8:version u64:seed u8:factions u8:ai_mask u8:teams(2-bit packed)
//!   u8:my_faction(255 = observer) u8:versus u32:final_step
//!   then records until EOF: u32:step u8:faction <cmd, wire-encoded>

use crate::net::{dec_cmd, enc_cmd, pack_teams, unpack_teams, Buf, Rdr};
use crate::world::{Cmd, MAX_FACTIONS};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Where finished matches are saved, relative to the working directory.
pub const DIR: &str = "replays";
const MAGIC: &[u8; 4] = b"NREP";
/// Bump whenever the sim's interpretation of a command changes, not just the
/// file layout. An old recording replayed under new rules silently diverges.
const VERSION: u8 = 2;
/// Matches shorter than this many steps, ten seconds, are not saved, so an
/// abandoned match does not litter the replay list.
const MIN_STEPS: u32 = 600;

/// The static match configuration a replay needs to rebuild the identical
/// world, plus how long the recording ran.
#[derive(Clone, Copy)]
pub struct Info {
    pub seed: u64,
    pub factions: usize,
    pub is_ai: [bool; MAX_FACTIONS],
    pub alliance: [u8; MAX_FACTIONS],
    /// The recorder's faction index. Playback shows their fog by default.
    /// A value >= MAX_FACTIONS means the recording came from an observer.
    pub my_faction: usize,
    pub versus: bool,
    pub final_step: u32,
}

impl Info {
    /// Recording length as (minutes, seconds) of game time.
    pub fn duration(&self) -> (u32, u32) {
        let s = self.final_step / 60;
        (s / 60, s % 60)
    }
    /// Short mode tag for filenames and the replay list.
    pub fn mode(&self) -> &'static str {
        if self.my_faction >= MAX_FACTIONS {
            "obs"
        } else if self.versus {
            "mp"
        } else {
            "sp"
        }
    }
}

/// A loaded replay: the match config and every command in the order it was
/// applied, each stamped with its sim step.
pub struct Replay {
    pub info: Info,
    pub cmds: Vec<(u32, u8, Cmd)>,
}

/// Accumulates a live match's commands; `finish` writes the file.
pub struct Recorder {
    info: Info,
    body: Buf,
}

impl Recorder {
    pub fn new(
        seed: u64,
        factions: usize,
        is_ai: [bool; MAX_FACTIONS],
        alliance: [u8; MAX_FACTIONS],
        my_faction: usize,
        versus: bool,
    ) -> Recorder {
        Recorder {
            info: Info { seed, factions, is_ai, alliance, my_faction, versus, final_step: 0 },
            body: Buf::default(),
        }
    }

    /// Log one command as applied to the sim at `step` by `faction`. Call in
    /// application order: playback re-applies records in file order.
    pub fn record(&mut self, step: u32, faction: usize, cmd: &Cmd) {
        self.body.u32(step);
        self.body.u8(faction as u8);
        enc_cmd(&mut self.body, cmd);
    }

    /// Write the recording of a match that ran `final_step` sim steps to the
    /// replay directory. Returns the saved path, or None for matches too
    /// short to keep or if the filesystem refuses.
    pub fn finish(self, final_step: u32) -> Option<PathBuf> {
        self.finish_in(Path::new(DIR), final_step)
    }

    fn finish_in(mut self, dir: &Path, final_step: u32) -> Option<PathBuf> {
        if final_step < MIN_STEPS {
            return None;
        }
        self.info.final_step = final_step;
        let bytes = encode(&self.info, &self.body.0);
        std::fs::create_dir_all(dir).ok()?;
        // Timestamped name; on a same-second collision, bump a suffix.
        let stamp = timestamp();
        let mut path = dir.join(format!("nanorts-{}-{}.nrep", stamp, self.info.mode()));
        let mut n = 2;
        while path.exists() {
            path = dir.join(format!("nanorts-{}-{}-{}.nrep", stamp, self.info.mode(), n));
            n += 1;
        }
        match std::fs::write(&path, bytes) {
            Ok(()) => Some(path),
            Err(e) => {
                eprintln!("could not save replay {}: {e}", path.display());
                None
            }
        }
    }
}

fn encode(info: &Info, body: &[u8]) -> Vec<u8> {
    let mut b = Buf::default();
    b.0.extend_from_slice(MAGIC);
    b.u8(VERSION);
    b.u64(info.seed);
    b.u8(info.factions as u8);
    let ai_mask = (0..MAX_FACTIONS).fold(0u8, |m, f| m | ((info.is_ai[f] as u8) << f));
    b.u8(ai_mask);
    b.u8(pack_teams(&info.alliance));
    b.u8(info.my_faction.min(255) as u8);
    b.u8(info.versus as u8);
    b.u32(info.final_step);
    b.0.extend_from_slice(body);
    b.0
}

fn parse_info(r: &mut Rdr) -> Option<Info> {
    for &m in MAGIC.iter() {
        if r.u8()? != m {
            return None;
        }
    }
    if r.u8()? != VERSION {
        return None;
    }
    let seed = r.u64()?;
    let factions = r.u8()? as usize;
    if !(2..=MAX_FACTIONS).contains(&factions) {
        return None;
    }
    let ai_mask = r.u8()?;
    let mut is_ai = [false; MAX_FACTIONS];
    for (f, slot) in is_ai.iter_mut().enumerate() {
        *slot = ai_mask & (1 << f) != 0;
    }
    let alliance = unpack_teams(r.u8()?);
    let my_faction = r.u8()? as usize;
    // A viewer slot must be one of the match's factions: the fog grids only
    // cover those.
    if (factions..MAX_FACTIONS).contains(&my_faction) {
        return None;
    }
    let versus = r.u8()? != 0;
    let final_step = r.u32()?;
    Some(Info { seed, factions, is_ai, alliance, my_faction, versus, final_step })
}

fn parse(bytes: &[u8]) -> Option<Replay> {
    let mut r = Rdr::new(bytes);
    let info = parse_info(&mut r)?;
    let mut cmds = Vec::new();
    while !r.done() {
        let step = r.u32()?;
        let faction = r.u8()?;
        cmds.push((step, faction, dec_cmd(&mut r)?));
    }
    Some(Replay { info, cmds })
}

/// Load a full replay from disk.
pub fn load(path: &Path) -> io::Result<Replay> {
    let bytes = std::fs::read(path)?;
    parse(&bytes).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not a nanorts replay"))
}

/// Read just the header, enough for the replay list to label an entry.
pub fn peek(path: &Path) -> io::Result<Info> {
    let bytes = std::fs::read(path)?;
    parse_info(&mut Rdr::new(&bytes))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not a nanorts replay"))
}

/// Every readable replay on disk, newest first. The timestamped filenames
/// sort chronologically.
pub fn list() -> Vec<(PathBuf, Info)> {
    let Ok(dir) = std::fs::read_dir(DIR) else {
        return Vec::new();
    };
    let mut v: Vec<(PathBuf, Info)> = dir
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "nrep"))
        .filter_map(|p| peek(&p).ok().map(|i| (p, i)))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v
}

/// `YYYYMMDD-HHMMSS` in UTC, from the system clock alone.
fn timestamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86400) as i64);
    let tod = secs % 86400;
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m, d, tod / 3600, (tod % 3600) / 60, tod % 60)
}

/// Days-since-epoch to (year, month, day). Howard Hinnant's civil_from_days.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Kind, Team, World};

    /// Encode a recording, parse it back, and check the header and every
    /// command survive byte-exact.
    #[test]
    fn file_round_trips() {
        let mut rec = Recorder::new(0xDEAD_BEEF, 3, [false, true, true, false], [0, 1, 2, 3], 0, false);
        rec.record(12, 0, &Cmd::Train { building: 7, unit: Kind::Worker });
        rec.record(90, 0, &Cmd::Order { ids: vec![3, 4, 5], x: 100.0, y: 220.5, attack_move: true, queue: false });
        rec.record(90, 2, &Cmd::Surrender);
        rec.info.final_step = 1234;
        let bytes = encode(&rec.info, &rec.body.0);
        let rp = parse(&bytes).expect("parses");
        assert_eq!(rp.info.seed, 0xDEAD_BEEF);
        assert_eq!(rp.info.factions, 3);
        assert_eq!(rp.info.is_ai, [false, true, true, false]);
        assert_eq!(rp.info.alliance, [0, 1, 2, 3]);
        assert_eq!(rp.info.final_step, 1234);
        assert!(!rp.info.versus);
        assert_eq!(rp.cmds.len(), 3);
        assert_eq!(rp.cmds[0].0, 12);
        assert!(matches!(rp.cmds[0].2, Cmd::Train { building: 7, unit: Kind::Worker }));
        assert_eq!(rp.cmds[1].1, 0);
        match &rp.cmds[1].2 {
            Cmd::Order { ids, x, y, attack_move, queue } => {
                assert_eq!(ids, &vec![3, 4, 5]);
                assert_eq!((*x, *y), (100.0, 220.5));
                assert!(*attack_move && !*queue);
            }
            other => panic!("wrong cmd: {:?}", other),
        }
        assert!(matches!(rp.cmds[2], (90, 2, Cmd::Surrender)));
    }

    /// The disk lifecycle: finish writes a file that load reads back, and a
    /// too-short match writes nothing.
    #[test]
    fn finish_writes_a_loadable_file() {
        let dir = std::env::temp_dir().join(format!("nanorts-replay-test-{}", std::process::id()));
        let mut rec = Recorder::new(9, 2, [false, true, false, false], [0, 1, 2, 3], 0, false);
        rec.record(5, 0, &Cmd::Surrender);
        let path = rec.finish_in(&dir, 6000).expect("a real match saves");
        let rp = load(&path).expect("and loads back");
        assert_eq!(rp.info.final_step, 6000);
        assert_eq!(rp.cmds.len(), 1);
        let stub = Recorder::new(9, 2, [false, true, false, false], [0, 1, 2, 3], 0, false);
        assert!(stub.finish_in(&dir, 60).is_none(), "abandoned lobbies save nothing");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Truncated or corrupt files must fail to parse, not panic or misread.
    #[test]
    fn rejects_garbage() {
        assert!(parse(b"").is_none());
        assert!(parse(b"NOPE").is_none());
        let rec = Recorder::new(1, 2, [false; 4], [0, 1, 2, 3], 0, false);
        let mut bytes = encode(&rec.info, &rec.body.0);
        bytes.push(9); // a dangling half-record
        assert!(parse(&bytes).is_none());
        // A viewer slot outside the match's factions.
        let rec = Recorder::new(1, 2, [false; 4], [0, 1, 2, 3], 3, false);
        assert!(parse(&encode(&rec.info, &rec.body.0)).is_none());
    }

    /// The core replay guarantee: re-running a recorded match from its Info,
    /// applying each command at its recorded step, reproduces the original
    /// simulation checksum-exact.
    #[test]
    fn playback_reproduces_the_sim() {
        let seed = 0xC0FFEE;
        let dt = 1.0 / 60.0;
        let is_ai = [false, true, false, false];
        let mut rec = Recorder::new(seed, 2, is_ai, [0, 1, 2, 3], 0, false);

        // Live game: a human faction issues commands at various steps while
        // the AI plays; record exactly what we apply.
        let mut live = World::new_match(seed, 2, is_ai, Team::Player, false);
        for _ in 0..1200 {
            if live.step == 30 {
                if let Some(b) = live.first_base(Team::Player) {
                    let c = Cmd::Train { building: live.ents[b].id, unit: Kind::Worker };
                    rec.record(live.step, 0, &c);
                    live.apply_cmd(Team::Player, &c);
                }
            }
            if live.step == 300 {
                let ids: Vec<u32> = live
                    .ents
                    .iter()
                    .filter(|e| e.team == Team::Player && e.kind == Kind::Worker)
                    .map(|e| e.id)
                    .collect();
                let c = Cmd::Order { ids, x: live.world_w * 0.4, y: live.world_h * 0.5, attack_move: true, queue: false };
                rec.record(live.step, 0, &c);
                live.apply_cmd(Team::Player, &c);
            }
            live.update(dt);
        }

        // Playback: rebuild the world from the recording and feed the file.
        rec.info.final_step = live.step;
        let rp = parse(&encode(&rec.info, &rec.body.0)).expect("parses");
        let mut back = World::new_match_teams(
            rp.info.seed,
            rp.info.factions,
            rp.info.is_ai,
            Team::from_idx(rp.info.my_faction),
            rp.info.versus,
            rp.info.alliance,
        );
        let mut next = 0;
        while back.step < rp.info.final_step {
            while next < rp.cmds.len() && rp.cmds[next].0 <= back.step {
                let (_, f, c) = &rp.cmds[next];
                back.apply_cmd(Team::from_idx(*f as usize), c);
                next += 1;
            }
            back.update(dt);
        }
        assert_eq!(next, rp.cmds.len(), "every recorded command was replayed");
        assert_eq!(live.checksum(), back.checksum(), "replay must reproduce the sim exactly");
        assert_eq!(live.stats.len(), back.stats.len(), "the stats series replays too");
    }
}
