//! Serverless peer-to-peer netcode. The two machines never exchange game state
//! — only commands — because they run the identical deterministic simulation in
//! lockstep. One peer hosts (listens), the other joins (connects). Commands
//! issued now are scheduled a few steps in the future (input delay) and both
//! sides advance a step only once they hold both players' commands for it. A
//! periodic state checksum is swapped to catch any divergence.

use crate::world::{kind_from_u8, Cmd};
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::time::Duration;

pub const INPUT_DELAY: u32 = 3;
pub const CHECK_EVERY: u32 = 20;
/// Hard ceiling on a single length-prefixed frame. A Step carries at most ~255
/// commands; anything larger is corruption or an attack, so we drop the peer.
const MAX_FRAME: usize = 1 << 16; // 64 KiB

/// Fixed UDP port the host beacons on so joiners can find it with zero config.
pub const DISCO_PORT: u16 = 50321;
const PING_MAGIC: &[u8] = b"NANORTS-WHERE?";
const PONG_MAGIC: &[u8] = b"NANORTS-HERE!"; // followed by 2-byte LE tcp port

// ---- tiny byte writer/reader ----------------------------------------------

#[derive(Default)]
struct Buf(Vec<u8>);
impl Buf {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
}

struct Rdr<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rdr<'a> {
    fn new(b: &'a [u8]) -> Rdr<'a> {
        Rdr { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.p + n <= self.b.len() {
            let s = &self.b[self.p..self.p + n];
            self.p += n;
            Some(s)
        } else {
            None
        }
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
    fn f32(&mut self) -> Option<f32> {
        self.take(4).map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}

// ---- command (de)serialization --------------------------------------------

fn enc_ids(buf: &mut Buf, ids: &[u32]) {
    buf.u16(ids.len().min(u16::MAX as usize) as u16);
    for &id in ids.iter().take(u16::MAX as usize) {
        buf.u32(id);
    }
}
fn dec_ids(r: &mut Rdr) -> Option<Vec<u32>> {
    let n = r.u16()? as usize;
    // The claimed count can't be trusted until the ids are actually read, so
    // pre-allocate no more than the bytes remaining could possibly hold.
    let mut v = Vec::with_capacity(n.min((r.b.len() - r.p) / 4));
    for _ in 0..n {
        v.push(r.u32()?);
    }
    Some(v)
}

fn enc_cmd(buf: &mut Buf, c: &Cmd) {
    match c {
        Cmd::Order { ids, x, y, attack_move, queue } => {
            buf.u8(1);
            enc_ids(buf, ids);
            buf.f32(*x);
            buf.f32(*y);
            buf.u8(*attack_move as u8);
            buf.u8(*queue as u8);
        }
        Cmd::Stop { ids } => {
            buf.u8(2);
            enc_ids(buf, ids);
        }
        Cmd::Train { building, unit } => {
            buf.u8(3);
            buf.u32(*building);
            buf.u8(*unit as u8);
        }
        Cmd::Build { worker, kind, x, y, chain } => {
            buf.u8(4);
            buf.u32(*worker);
            buf.u8(*kind as u8);
            buf.f32(*x);
            buf.f32(*y);
            buf.u8(*chain as u8);
        }
        Cmd::Rally { building, x, y } => {
            buf.u8(5);
            buf.u32(*building);
            buf.f32(*x);
            buf.f32(*y);
        }
        Cmd::Cancel { building } => {
            buf.u8(6);
            buf.u32(*building);
        }
        Cmd::Surrender => {
            buf.u8(7); // no payload — the faction comes from the Step header
        }
    }
}
fn dec_cmd(r: &mut Rdr) -> Option<Cmd> {
    Some(match r.u8()? {
        1 => Cmd::Order {
            ids: dec_ids(r)?,
            x: r.f32()?,
            y: r.f32()?,
            attack_move: r.u8()? != 0,
            queue: r.u8()? != 0,
        },
        2 => Cmd::Stop { ids: dec_ids(r)? },
        3 => Cmd::Train {
            building: r.u32()?,
            unit: kind_from_u8(r.u8()?),
        },
        4 => Cmd::Build {
            worker: r.u32()?,
            kind: kind_from_u8(r.u8()?),
            x: r.f32()?,
            y: r.f32()?,
            chain: r.u8()? != 0,
        },
        5 => Cmd::Rally {
            building: r.u32()?,
            x: r.f32()?,
            y: r.f32()?,
        },
        6 => Cmd::Cancel { building: r.u32()? },
        7 => Cmd::Surrender,
        _ => return None,
    })
}

// ---- messages -------------------------------------------------------------

/// Wire messages. Step/Check carry the originating faction so the host can fan
/// commands out to all peers; Welcome/Start set up the match during the lobby.
/// `teams` packs each faction's alliance id into 2 bits (faction f at bits
/// 2f..2f+1) — [0,1,2,3] is a free-for-all, [0,0,1,1] a 2v2.
enum Msg {
    Step { faction: u8, step: u32, cmds: Vec<Cmd> },
    Check { faction: u8, step: u32, hash: u64 },
    Welcome { faction: u8 },                                   // host -> joiner on accept
    Start { seed: u64, factions: u8, ai_mask: u8, teams: u8 }, // host -> joiner to begin
}

/// Pack per-faction alliance ids (each 0..4) into the Start message's byte.
pub fn pack_teams(alliance: &[u8; MAXF]) -> u8 {
    (0..MAXF).fold(0u8, |b, f| b | ((alliance[f] & 3) << (f * 2)))
}
fn unpack_teams(b: u8) -> [u8; MAXF] {
    let mut a = [0u8; MAXF];
    for (f, slot) in a.iter_mut().enumerate() {
        *slot = (b >> (f * 2)) & 3;
    }
    a
}

/// The free-for-all team byte (every faction its own alliance).
const TEAMS_FFA: u8 = 0b11_10_01_00;

fn enc_msg(m: &Msg) -> Vec<u8> {
    let mut b = Buf::default();
    match m {
        Msg::Step { faction, step, cmds } => {
            b.u8(1);
            b.u8(*faction);
            b.u32(*step);
            b.u8(cmds.len().min(255) as u8);
            for c in cmds.iter().take(255) {
                enc_cmd(&mut b, c);
            }
        }
        Msg::Check { faction, step, hash } => {
            b.u8(2);
            b.u8(*faction);
            b.u32(*step);
            b.u64(*hash);
        }
        Msg::Welcome { faction } => {
            b.u8(3);
            b.u8(*faction);
        }
        Msg::Start { seed, factions, ai_mask, teams } => {
            b.u8(4);
            b.u64(*seed);
            b.u8(*factions);
            b.u8(*ai_mask);
            b.u8(*teams);
        }
    }
    b.0
}
fn dec_msg(bytes: &[u8]) -> Option<Msg> {
    let mut r = Rdr::new(bytes);
    Some(match r.u8()? {
        1 => {
            let faction = r.u8()?;
            let step = r.u32()?;
            let n = r.u8()? as usize;
            let mut cmds = Vec::with_capacity(n);
            for _ in 0..n {
                cmds.push(dec_cmd(&mut r)?);
            }
            Msg::Step { faction, step, cmds }
        }
        2 => Msg::Check {
            faction: r.u8()?,
            step: r.u32()?,
            hash: r.u64()?,
        },
        3 => Msg::Welcome { faction: r.u8()? },
        4 => Msg::Start {
            seed: r.u64()?,
            factions: r.u8()?,
            ai_mask: r.u8()?,
            teams: r.u8()?,
        },
        _ => return None,
    })
}

/// Sender-side ceilings for one Step. The wire format caps a step at 255
/// commands and an id list at a u16 count, and the frame reader drops any link
/// that sends a frame over MAX_FRAME — so an unclamped burst (commands piling
/// up while the sim is stalled) would either be silently truncated by the
/// encoder or kill the connection. Crucially the sender must clamp BEFORE
/// storing its own copy of the step: every sim has to consume the identical
/// list, and truncating only on the wire would desync the sender from
/// everyone else.
const STEP_MAX_CMDS: usize = 255;
/// Far beyond any real selection, yet small enough that even a single
/// max-size command still fits comfortably inside one frame.
const STEP_MAX_IDS: usize = 4096;
const STEP_MAX_BYTES: usize = MAX_FRAME / 2; // headroom under the frame limit

fn clamp_step(cmds: &mut Vec<Cmd>) {
    let mut total = 7; // Step header: tag + faction + step + count
    let mut keep = 0;
    while keep < cmds.len() && keep < STEP_MAX_CMDS {
        match &mut cmds[keep] {
            Cmd::Order { ids, .. } | Cmd::Stop { ids } => ids.truncate(STEP_MAX_IDS),
            _ => {}
        }
        let mut b = Buf::default();
        enc_cmd(&mut b, &cmds[keep]);
        if total + b.0.len() > STEP_MAX_BYTES {
            break;
        }
        total += b.0.len();
        keep += 1;
    }
    cmds.truncate(keep);
}

// ---- the transport --------------------------------------------------------

/// Cap on the userspace send queue. It only grows while the kernel buffer is
/// full; a peer that hasn't drained a megabyte of lockstep traffic is gone.
const MAX_OUTBUF: usize = 1 << 20; // 1 MiB

struct Peer {
    stream: TcpStream,
    /// The faction this link plays. Assigned once at accept time and kept for
    /// the peer's whole life — a joiner learns it from Welcome and never hears
    /// about it again, so the host must never renumber. A joiner's single peer
    /// is the host itself (faction 0).
    faction: usize,
    inbuf: Vec<u8>,
    outbuf: Vec<u8>,
    closed: bool,
}
impl Peer {
    fn new(stream: TcpStream, faction: usize) -> Peer {
        Peer { stream, faction, inbuf: Vec::new(), outbuf: Vec::new(), closed: false }
    }
    /// Queue a frame and push as much backlog as the socket will take. The
    /// stream is non-blocking, so a direct write_all would misreport a full
    /// kernel buffer as a dead link — and a partial write would leave the
    /// remote stuck on half a frame. All sends go through the queue instead.
    fn send(&mut self, m: &Msg) {
        if self.closed {
            return;
        }
        let bytes = enc_msg(m);
        if self.outbuf.len() + 4 + bytes.len() > MAX_OUTBUF {
            self.closed = true;
            return;
        }
        self.outbuf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        self.outbuf.extend_from_slice(&bytes);
        self.flush();
    }
    fn flush(&mut self) {
        while !self.outbuf.is_empty() {
            match self.stream.write(&self.outbuf) {
                Ok(0) => {
                    self.closed = true;
                    return;
                }
                Ok(n) => {
                    self.outbuf.drain(..n);
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => return,
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(_) => {
                    self.closed = true;
                    return;
                }
            }
        }
    }
    fn poll(&mut self) -> Vec<Msg> {
        self.flush();
        let mut tmp = [0u8; 8192];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => {
                    self.closed = true;
                    break;
                }
                Ok(n) => self.inbuf.extend_from_slice(&tmp[..n]),
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(_) => {
                    self.closed = true;
                    break;
                }
            }
        }
        let mut out = Vec::new();
        loop {
            if self.inbuf.len() < 4 {
                break;
            }
            let len = u32::from_le_bytes([self.inbuf[0], self.inbuf[1], self.inbuf[2], self.inbuf[3]]) as usize;
            // A real frame is tiny; a bogus length means a corrupt or hostile
            // peer — drop the link rather than buffer gigabytes (or overflow
            // `4 + len` on a 32-bit usize).
            if len > MAX_FRAME {
                self.closed = true;
                self.inbuf.clear();
                break;
            }
            if self.inbuf.len() < 4 + len {
                break;
            }
            let frame: Vec<u8> = self.inbuf[4..4 + len].to_vec();
            self.inbuf.drain(0..4 + len);
            if let Some(m) = dec_msg(&frame) {
                out.push(m);
            }
        }
        out
    }
}

const MAXF: usize = 4; // mirrors world::MAX_FACTIONS

/// Lockstep peers stay within INPUT_DELAY of each other, so a Step or Check
/// claiming a step this far past what we've sent is garbage; ignoring it keeps
/// the per-step maps bounded no matter what a hostile peer streams at us.
const MAX_AHEAD: u32 = 1200; // ~20s at 60 Hz

/// Send a single length-prefixed message on a blocking stream (handshake only).
fn send_framed(stream: &mut TcpStream, m: &Msg) -> std::io::Result<()> {
    let bytes = enc_msg(m);
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)
}
/// Read a single length-prefixed message from a blocking stream (handshake only).
fn read_framed(stream: &mut TcpStream) -> std::io::Result<Msg> {
    let mut lenb = [0u8; 4];
    stream.read_exact(&mut lenb)?;
    let len = u32::from_le_bytes(lenb) as usize;
    if len > MAX_FRAME {
        return Err(std::io::Error::new(ErrorKind::InvalidData, "frame too large"));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    dec_msg(&buf).ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "bad handshake message"))
}
fn connect_to(addr: &str) -> std::io::Result<TcpStream> {
    let budget = Duration::from_secs(6);
    let stream = match addr.parse() {
        Ok(sa) => TcpStream::connect_timeout(&sa, budget)?,
        Err(_) => {
            // A hostname: resolve it and split the budget across the results
            // so a dead host can't hang us for the OS default (minutes).
            // Resolution itself may still block; that's the DNS tax.
            let addrs: Vec<_> = addr.to_socket_addrs()?.collect();
            if addrs.is_empty() {
                return Err(std::io::Error::new(ErrorKind::InvalidInput, "address resolved to nothing"));
            }
            let per = budget / addrs.len() as u32;
            let mut last = None;
            let mut stream = None;
            for sa in &addrs {
                match TcpStream::connect_timeout(sa, per) {
                    Ok(s) => {
                        stream = Some(s);
                        break;
                    }
                    Err(e) => last = Some(e),
                }
            }
            match stream {
                Some(s) => s,
                None => return Err(last.unwrap()),
            }
        }
    };
    stream.set_nodelay(true).ok();
    Ok(stream)
}

/// Drives N-faction lockstep over a star of TCP links. Human factions exchange
/// commands (the host relays everyone's to everyone); AI factions are simulated
/// locally on every peer, so they cost no traffic.
pub struct Lockstep {
    peers: Vec<Peer>, // host: one per joiner; joiner: just the host
    is_host: bool,
    pub seed: u64,
    pub factions: usize,
    pub is_ai: [bool; MAXF],
    /// Alliance id per faction (from the host's Start message).
    pub alliance: [u8; MAXF],
    pub my_faction: usize,
    pub sim_step: u32,
    sent_step: u32,
    cmds: Vec<HashMap<u32, Vec<Cmd>>>, // [faction][step] -> commands
    local_checks: HashMap<u32, u64>,
    remote_checks: HashMap<(u32, u8), u64>, // (step, faction) -> hash
    pub pending: Vec<Cmd>,
    pub desync_step: Option<u32>,
    pub last_synced: u32,
    pub disconnected: bool,
}

impl Lockstep {
    fn new(
        peers: Vec<Peer>,
        is_host: bool,
        seed: u64,
        factions: usize,
        is_ai: [bool; MAXF],
        alliance: [u8; MAXF],
        my_faction: usize,
    ) -> Lockstep {
        Lockstep {
            peers,
            is_host,
            seed,
            factions,
            is_ai,
            alliance,
            my_faction,
            sim_step: 0,
            sent_step: 0,
            cmds: vec![HashMap::new(); MAXF],
            local_checks: HashMap::new(),
            remote_checks: HashMap::new(),
            pending: Vec::new(),
            desync_step: None,
            last_synced: 0,
            disconnected: false,
        }
    }

    /// Blocking 2-player host (faction 0). Used by the headless determinism test.
    pub fn host(port: u16, seed: u64) -> std::io::Result<Lockstep> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let (mut stream, _addr) = listener.accept()?;
        stream.set_nodelay(true).ok();
        send_framed(&mut stream, &Msg::Welcome { faction: 1 })?;
        send_framed(&mut stream, &Msg::Start { seed, factions: 2, ai_mask: 0, teams: TEAMS_FFA })?;
        stream.set_nonblocking(true)?;
        Ok(Lockstep::new(vec![Peer::new(stream, 1)], true, seed, 2, [false; MAXF], unpack_teams(TEAMS_FFA), 0))
    }

    /// Blocking 2-player joiner (faction 1). Used by the headless test.
    pub fn join(addr: &str) -> std::io::Result<Lockstep> {
        let mut stream = connect_to(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(6))).ok();
        // The handshake values index fixed-size faction tables everywhere
        // downstream, so anything out of range is a protocol error here.
        let my_faction = match read_framed(&mut stream)? {
            Msg::Welcome { faction } if (1..MAXF as u8).contains(&faction) => faction as usize,
            _ => return Err(std::io::Error::new(ErrorKind::InvalidData, "bad welcome")),
        };
        let (seed, factions, ai_mask, teams) = match read_framed(&mut stream)? {
            Msg::Start { seed, factions, ai_mask, teams } if (2..=MAXF as u8).contains(&factions) => {
                (seed, factions as usize, ai_mask, teams)
            }
            _ => return Err(std::io::Error::new(ErrorKind::InvalidData, "bad start")),
        };
        if my_faction >= factions {
            return Err(std::io::Error::new(ErrorKind::InvalidData, "faction out of range"));
        }
        stream.set_nonblocking(true)?;
        let mut is_ai = [false; MAXF];
        for (f, slot) in is_ai.iter_mut().enumerate() {
            *slot = ai_mask & (1 << f) != 0;
        }
        Ok(Lockstep::new(vec![Peer::new(stream, 0)], false, seed, factions, is_ai, unpack_teams(teams), my_faction))
    }

    pub fn alive(&self) -> bool {
        !self.disconnected
    }

    pub fn queue(&mut self, c: Cmd) {
        self.pending.push(c);
    }

    fn relay(&mut self, from: usize, m: &Msg) {
        for (k, p) in self.peers.iter_mut().enumerate() {
            if k != from {
                p.send(m);
            }
        }
    }

    /// Receive everyone's commands (host fans them out), then push our own ahead.
    pub fn service(&mut self) {
        for j in 0..self.peers.len() {
            for m in self.peers[j].poll() {
                self.ingest(j, m);
            }
        }
        while self.sent_step < self.sim_step + INPUT_DELAY {
            let mut cmds = std::mem::take(&mut self.pending);
            // Clamp before the local copy is stored: every sim (ours included)
            // must consume exactly the list that goes on the wire.
            clamp_step(&mut cmds);
            self.cmds[self.my_faction].insert(self.sent_step, cmds.clone());
            let msg = Msg::Step { faction: self.my_faction as u8, step: self.sent_step, cmds };
            for p in &mut self.peers {
                p.send(&msg);
            }
            self.sent_step += 1;
        }
        if self.peers.iter().any(|p| p.closed) {
            self.disconnected = true;
        }
    }

    /// Store one in-game message from peer `from`. Both the per-frame service
    /// loop and the lobby-start backlog (Steps that arrived in the same poll
    /// batch as Start) funnel through here; the host additionally fans each
    /// message out to every other peer.
    fn ingest(&mut self, from: usize, m: Msg) {
        match m {
            Msg::Step { faction, step, cmds } => {
                let f = faction as usize;
                // The host knows which faction each link plays; a Step
                // claiming any other faction is spoofed — drop it unrelayed.
                if self.is_host && self.peers.get(from).map_or(true, |p| p.faction != f) {
                    return;
                }
                if step > self.sent_step.saturating_add(MAX_AHEAD) {
                    return;
                }
                if self.is_host {
                    self.relay(from, &Msg::Step { faction, step, cmds: cmds.clone() });
                }
                if f < self.cmds.len() {
                    self.cmds[f].entry(step).or_insert(cmds);
                }
            }
            Msg::Check { faction, step, hash } => {
                if self.is_host && self.peers.get(from).map_or(true, |p| p.faction != faction as usize) {
                    return;
                }
                if self.is_host {
                    self.relay(from, &Msg::Check { faction, step, hash });
                }
                // Once a desync is flagged the check maps are frozen as
                // evidence — stop accumulating.
                if self.desync_step.is_none() && step <= self.sent_step.saturating_add(MAX_AHEAD) {
                    self.remote_checks.insert((step, faction), hash);
                }
            }
            _ => {} // Welcome/Start are lobby-only
        }
    }

    /// Can we simulate the next step (do we hold every HUMAN faction's input)?
    pub fn ready(&self) -> bool {
        for f in 0..self.factions {
            if !self.is_ai[f] && !self.cmds[f].contains_key(&self.sim_step) {
                return false;
            }
        }
        true
    }

    /// Each faction's commands for the current step (AI factions are empty —
    /// their orders come from the locally-run AI inside `World::update`).
    pub fn step_cmds(&self) -> Vec<Vec<Cmd>> {
        (0..self.factions)
            .map(|f| self.cmds[f].get(&self.sim_step).cloned().unwrap_or_default())
            .collect()
    }

    /// Record + broadcast the post-step checksum, advance, and flag any desync.
    pub fn advanced(&mut self, checksum: u64) {
        if self.sim_step % CHECK_EVERY == 0 {
            self.local_checks.insert(self.sim_step, checksum);
            let msg = Msg::Check { faction: self.my_faction as u8, step: self.sim_step, hash: checksum };
            for p in &mut self.peers {
                p.send(&msg);
            }
        }
        self.sim_step += 1;
        // A step is settled only once EVERY human remote faction has reported
        // a matching hash for it; a single mismatch anywhere is a desync.
        // After a desync the match is already lost — stop re-scanning and
        // leave the maps frozen as evidence.
        if self.desync_step.is_none() {
            let mut settled: Vec<u32> = Vec::new();
            for (&s, &lh) in &self.local_checks {
                let mut complete = true;
                for f in 0..self.factions {
                    if f == self.my_faction || self.is_ai[f] {
                        continue;
                    }
                    match self.remote_checks.get(&(s, f as u8)) {
                        Some(&rh) if rh == lh => {}
                        Some(_) => {
                            complete = false;
                            self.desync_step = Some(self.desync_step.map_or(s, |d| d.min(s)));
                        }
                        None => complete = false,
                    }
                }
                if complete {
                    self.last_synced = self.last_synced.max(s);
                    settled.push(s);
                }
            }
            for s in settled {
                self.local_checks.remove(&s);
                for f in 0..self.factions {
                    self.remote_checks.remove(&(s, f as u8));
                }
            }
        }
        let cutoff = self.sim_step;
        for f in 0..self.factions {
            self.cmds[f].retain(|&s, _| s >= cutoff);
        }
    }
}

// ---- host lobby: accept up to `factions-1` joiners + LAN beacon ----------

/// A listening host gathering players in the lobby. Pumped each frame: it
/// answers LAN discovery pings and accepts joiners (assigning each the next
/// faction), without ever blocking the render loop.
pub struct Host {
    listener: TcpListener,
    beacon: Option<UdpSocket>,
    pub port: u16,
    seed: u64,
    pub factions: usize,
    /// Alliance id per faction — the host picks the mode in the lobby
    /// ([0,1,2,3] FFA, [0,0,1,1] 2v2) and broadcasts it in Start.
    pub teams: [u8; MAXF],
    peers: Vec<Peer>, // each carries the faction it was welcomed with
}

impl Host {
    pub fn bind(port: u16, seed: u64, factions: usize) -> std::io::Result<Host> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        listener.set_nonblocking(true)?;
        let beacon = UdpSocket::bind(("0.0.0.0", DISCO_PORT)).ok().and_then(|s| {
            s.set_nonblocking(true).ok()?;
            Some(s)
        });
        Ok(Host {
            listener,
            beacon,
            port,
            seed,
            factions: factions.clamp(2, MAXF),
            teams: unpack_teams(TEAMS_FFA),
            peers: Vec::new(),
        })
    }

    /// Answer discovery pings and accept any waiting joiner (up to the player
    /// count). Each joiner is assigned the next faction and sent a Welcome.
    pub fn poll(&mut self) {
        if let Some(b) = &self.beacon {
            let mut buf = [0u8; 64];
            while let Ok((n, from)) = b.recv_from(&mut buf) {
                if &buf[..n] == PING_MAGIC {
                    let mut reply = PONG_MAGIC.to_vec();
                    reply.extend_from_slice(&self.port.to_le_bytes());
                    let _ = b.send_to(&reply, from);
                }
            }
        }
        // Drop any joiner that has bailed out before the game starts. The
        // survivors keep the factions they were welcomed with — each joiner
        // already believes its Welcome, so renumbering here would deadlock the
        // match on a faction nobody plays.
        for p in self.peers.iter_mut() {
            let _ = p.poll();
        }
        self.peers.retain(|p| !p.closed);

        // A new joiner takes the lowest faction no live peer holds, so a slot
        // freed by a leaver is refilled before the lobby grows past it.
        let free = (1..self.factions).find(|&f| self.peers.iter().all(|p| p.faction != f));
        if let Some(faction) = free {
            if let Ok((mut stream, _addr)) = self.listener.accept() {
                stream.set_nodelay(true).ok();
                if send_framed(&mut stream, &Msg::Welcome { faction: faction as u8 }).is_ok() && stream.set_nonblocking(true).is_ok() {
                    self.peers.push(Peer::new(stream, faction));
                }
            }
        } else {
            // Lobby full: accept and hang up so a latecomer sees the refusal
            // at once instead of idling in the backlog until its Welcome read
            // times out.
            while let Ok((stream, _addr)) = self.listener.accept() {
                drop(stream);
            }
        }
    }

    /// Players connected so far (including the host).
    pub fn players(&self) -> usize {
        self.peers.len() + 1
    }

    /// Begin the match: unfilled slots become AI; broadcast Start; build the link.
    pub fn start(mut self) -> Lockstep {
        let mut is_ai = [false; MAXF];
        for (f, slot) in is_ai.iter_mut().enumerate().take(self.factions) {
            *slot = f != 0 && self.peers.iter().all(|p| p.faction != f);
        }
        let ai_mask = (0..MAXF).fold(0u8, |m, f| if is_ai[f] { m | (1 << f) } else { m });
        let start = Msg::Start {
            seed: self.seed,
            factions: self.factions as u8,
            ai_mask,
            teams: pack_teams(&self.teams),
        };
        for p in &mut self.peers {
            p.send(&start);
        }
        Lockstep::new(self.peers, true, self.seed, self.factions, is_ai, self.teams, 0)
    }
}

// ---- joiner lobby: connect, learn our faction, wait for Start ------------

/// A connected joiner waiting in the host's lobby for the match to start.
pub struct Joiner {
    peer: Option<Peer>,
    pub my_faction: usize,
}

impl Joiner {
    /// Connect and read our assigned faction (blocking, bounded).
    pub fn connect(addr: &str) -> std::io::Result<Joiner> {
        let mut stream = connect_to(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(6))).ok();
        // Only 1..MAXF are joinable slots (0 is the host); anything else would
        // index out of the fixed-size faction tables, so refuse the handshake.
        let my_faction = match read_framed(&mut stream)? {
            Msg::Welcome { faction } if (1..MAXF as u8).contains(&faction) => faction as usize,
            _ => return Err(std::io::Error::new(ErrorKind::InvalidData, "bad welcome")),
        };
        stream.set_read_timeout(None).ok();
        stream.set_nonblocking(true)?;
        Ok(Joiner { peer: Some(Peer::new(stream, 0)), my_faction })
    }

    /// Non-blocking: returns the live Lockstep once the host begins the match.
    pub fn poll_start(&mut self) -> Option<Lockstep> {
        // Drain this batch. The host sends Start before any Step, so a batch with
        // the host's first commands also carries Start — keep BOTH (we used to
        // drop the early steps, deadlocking peers that connected first).
        let msgs = self.peer.as_mut()?.poll();
        let mut cfg = None;
        let mut backlog = Vec::new();
        for m in msgs {
            match m {
                Msg::Start { seed, factions, ai_mask, teams } if cfg.is_none() => {
                    cfg = Some((seed, factions, ai_mask, teams))
                }
                other => backlog.push(other),
            }
        }
        let (seed, factions, ai_mask, teams) = cfg?;
        let factions = factions as usize;
        // A Start describing an impossible match would blow past the
        // fixed-size faction tables later; treat it as a protocol violation
        // and hang up rather than build a Lockstep from it.
        if !(2..=MAXF).contains(&factions) || self.my_faction >= factions {
            if let Some(p) = self.peer.as_mut() {
                p.closed = true;
            }
            return None;
        }
        let peer = self.peer.take()?;
        let mut is_ai = [false; MAXF];
        for (f, slot) in is_ai.iter_mut().enumerate() {
            *slot = ai_mask & (1 << f) != 0;
        }
        let mut ls = Lockstep::new(vec![peer], false, seed, factions, is_ai, unpack_teams(teams), self.my_faction);
        for m in backlog {
            ls.ingest(0, m);
        }
        Some(ls)
    }

    pub fn alive(&self) -> bool {
        self.peer.as_ref().map(|p| !p.closed).unwrap_or(true)
    }
}

// ---- joiner-side LAN discovery -------------------------------------------

/// Broadcasts "where are you?" pings and listens for a host's reply.
pub struct Discover {
    sock: UdpSocket,
}

impl Discover {
    pub fn start() -> std::io::Result<Discover> {
        let sock = UdpSocket::bind(("0.0.0.0", 0))?;
        sock.set_nonblocking(true)?;
        sock.set_broadcast(true)?;
        Ok(Discover { sock })
    }

    /// Fire a discovery ping to the whole subnet.
    pub fn ping(&self) {
        let _ = self
            .sock
            .send_to(PING_MAGIC, (Ipv4Addr::BROADCAST, DISCO_PORT));
    }

    /// Non-blocking: returns a host's address if one just answered.
    pub fn poll(&self) -> Option<SocketAddrV4> {
        let mut buf = [0u8; 64];
        loop {
            match self.sock.recv_from(&mut buf) {
                Ok((n, from)) => {
                    if n >= PONG_MAGIC.len() + 2 && &buf[..PONG_MAGIC.len()] == PONG_MAGIC {
                        let port = u16::from_le_bytes([buf[PONG_MAGIC.len()], buf[PONG_MAGIC.len() + 1]]);
                        if let std::net::IpAddr::V4(ip) = from.ip() {
                            return Some(SocketAddrV4::new(ip, port));
                        }
                    }
                }
                Err(_) => return None,
            }
        }
    }
}

/// Best-effort: our own LAN IPv4 (for display so a peer can join manually).
/// Works by asking the OS which local address routes toward a public IP — no
/// packets are actually sent for a connectionless UDP socket.
pub fn local_ipv4() -> Option<Ipv4Addr> {
    let sock = UdpSocket::bind(("0.0.0.0", 0)).ok()?;
    sock.connect(("8.8.8.8", 80)).ok()?;
    match sock.local_addr().ok()?.ip() {
        std::net::IpAddr::V4(ip) if !ip.is_unspecified() => Some(ip),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU16, Ordering};

    /// Each listening test takes a distinct loopback port so the suite can run
    /// in parallel. (The lobby's UDP beacon may fail to bind when two Host
    /// tests overlap; Host::bind treats that as optional, which is fine here.)
    static NEXT_PORT: AtomicU16 = AtomicU16::new(47431);
    fn test_port() -> u16 {
        NEXT_PORT.fetch_add(1, Ordering::Relaxed)
    }

    fn connect_bg(addr: String) -> std::thread::JoinHandle<std::io::Result<Joiner>> {
        std::thread::spawn(move || Joiner::connect(&addr))
    }

    /// Pump the lobby until it reaches `want` players (host included).
    fn pump(host: &mut Host, want: usize) {
        for _ in 0..4000 {
            host.poll();
            if host.players() == want {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("lobby never reached {want} players (got {})", host.players());
    }

    #[test]
    fn lobby_keeps_factions_across_leavers() {
        let port = test_port();
        let mut host = Host::bind(port, 7, 4).unwrap();
        let addr = format!("127.0.0.1:{port}");

        let t1 = connect_bg(addr.clone());
        pump(&mut host, 2);
        let j1 = t1.join().unwrap().unwrap();
        let t2 = connect_bg(addr.clone());
        pump(&mut host, 3);
        let j2 = t2.join().unwrap().unwrap();
        assert_eq!((j1.my_faction, j2.my_faction), (1, 2));

        // Joiner 1 bails: joiner 2 must KEEP faction 2 (it was already told
        // so), and the next arrival must take the vacated faction 1.
        drop(j1);
        pump(&mut host, 2);
        let t3 = connect_bg(addr);
        pump(&mut host, 3);
        let mut j3 = t3.join().unwrap().unwrap();
        assert_eq!(j3.my_faction, 1);

        // Start with the last slot unfilled: only faction 3 becomes AI, and
        // the joiners' Start-derived view agrees with the host's.
        let ls = host.start();
        assert_eq!(ls.is_ai, [false, false, false, true]);
        for _ in 0..4000 {
            if let Some(jls) = j3.poll_start() {
                assert_eq!(jls.is_ai, ls.is_ai);
                assert_eq!(jls.my_faction, 1);
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("joiner never saw Start");
    }

    #[test]
    fn full_lobby_hangs_up_latecomers() {
        let port = test_port();
        let mut host = Host::bind(port, 7, 2).unwrap(); // one joinable slot
        let addr = format!("127.0.0.1:{port}");
        let t1 = connect_bg(addr.clone());
        pump(&mut host, 2);
        let _j1 = t1.join().unwrap().unwrap();

        // A latecomer must be refused promptly (host closes the socket), not
        // parked in the backlog until its 6-second Welcome read expires.
        let t2 = connect_bg(addr);
        for _ in 0..4000 {
            host.poll();
            if t2.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(t2.is_finished(), "latecomer left waiting");
        assert!(t2.join().unwrap().is_err());
        assert_eq!(host.players(), 2);
    }

    #[test]
    fn handshake_rejects_bogus_values() {
        let port = test_port();
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();

        // A Welcome carrying an impossible faction must fail the connect.
        let t = connect_bg(format!("127.0.0.1:{port}"));
        let (mut s, _) = listener.accept().unwrap();
        send_framed(&mut s, &Msg::Welcome { faction: MAXF as u8 }).unwrap();
        assert!(t.join().unwrap().is_err());

        // A Start with an absurd faction count must kill the lobby link, not
        // hand back a Lockstep that would index out of the faction tables.
        let t = connect_bg(format!("127.0.0.1:{port}"));
        let (mut s, _) = listener.accept().unwrap();
        send_framed(&mut s, &Msg::Welcome { faction: 1 }).unwrap();
        let mut j = t.join().unwrap().unwrap();
        send_framed(&mut s, &Msg::Start { seed: 1, factions: (MAXF + 1) as u8, ai_mask: 0, teams: TEAMS_FFA }).unwrap();
        for _ in 0..4000 {
            assert!(j.poll_start().is_none(), "bogus start accepted");
            if !j.alive() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("bogus start neither accepted nor refused");
    }

    #[test]
    fn sender_clamps_before_storing() {
        // Oversized id lists shrink to the cap, excess commands drop from the
        // tail, and the encoded Step always fits well inside one frame.
        let mut cmds = vec![Cmd::Stop { ids: (0..100_000).collect() }];
        for i in 0..400u32 {
            cmds.push(Cmd::Cancel { building: i });
        }
        clamp_step(&mut cmds);
        match &cmds[0] {
            Cmd::Stop { ids } => assert_eq!(ids.len(), STEP_MAX_IDS),
            _ => panic!("head command dropped"),
        }
        assert_eq!(cmds.len(), STEP_MAX_CMDS);
        let frame = enc_msg(&Msg::Step { faction: 0, step: 0, cmds });
        assert!(frame.len() <= STEP_MAX_BYTES && frame.len() < MAX_FRAME);
    }

    #[test]
    fn team_bytes_roundtrip() {
        for teams in [[0u8, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 1], [2, 2, 2, 2]] {
            assert_eq!(unpack_teams(pack_teams(&teams)), teams);
        }
        assert_eq!(unpack_teams(TEAMS_FFA), [0, 1, 2, 3], "the FFA byte is everyone-for-themselves");
    }

    #[test]
    fn surrender_roundtrips_on_the_wire() {
        let mut b = Buf::default();
        enc_cmd(&mut b, &Cmd::Surrender);
        let mut r = Rdr::new(&b.0);
        assert!(matches!(dec_cmd(&mut r), Some(Cmd::Surrender)));
        assert_eq!(r.p, b.0.len(), "no stray payload bytes");
    }

    #[test]
    fn dec_ids_rejects_short_input() {
        // A count prefix claiming 65535 ids backed by 4 bytes must fail
        // cleanly (the pre-allocation is bounded by the bytes present).
        let mut b = Buf::default();
        b.u16(u16::MAX);
        b.u32(7);
        assert!(dec_ids(&mut Rdr::new(&b.0)).is_none());
    }

    #[test]
    fn checks_are_keyed_per_faction() {
        // 3 humans: faction 2's mismatching hash must flag a desync even when
        // faction 1's matching hash arrives afterwards (step-keyed storage
        // used to let the later Check overwrite the earlier one).
        let mut ls = Lockstep::new(Vec::new(), false, 1, 3, [false; MAXF], unpack_teams(TEAMS_FFA), 0);
        ls.advanced(0xABCD); // sim_step 0 records a local check
        ls.ingest(0, Msg::Check { faction: 2, step: 0, hash: 0xBEEF });
        ls.ingest(0, Msg::Check { faction: 1, step: 0, hash: 0xABCD });
        ls.advanced(1);
        assert_eq!(ls.desync_step, Some(0));
        assert_eq!(ls.last_synced, 0);
        // Once flagged, further checks are not accumulated.
        ls.ingest(0, Msg::Check { faction: 1, step: CHECK_EVERY, hash: 5 });
        assert!(!ls.remote_checks.contains_key(&(CHECK_EVERY, 1)));
    }

    #[test]
    fn last_synced_needs_every_faction() {
        let mut ls = Lockstep::new(Vec::new(), false, 1, 3, [false; MAXF], unpack_teams(TEAMS_FFA), 0);
        while ls.sim_step <= CHECK_EVERY {
            ls.advanced(0xABCD); // local checks land at steps 0 and CHECK_EVERY
        }
        ls.ingest(0, Msg::Check { faction: 1, step: 0, hash: 0xABCD });
        ls.ingest(0, Msg::Check { faction: 1, step: CHECK_EVERY, hash: 0xABCD });
        ls.advanced(0xABCD);
        assert_eq!(ls.last_synced, 0); // faction 2 hasn't reported yet
        ls.ingest(0, Msg::Check { faction: 2, step: 0, hash: 0xABCD });
        ls.ingest(0, Msg::Check { faction: 2, step: CHECK_EVERY, hash: 0xABCD });
        ls.advanced(0xABCD);
        assert_eq!(ls.last_synced, CHECK_EVERY);
        assert_eq!(ls.desync_step, None);
        assert!(ls.local_checks.is_empty() && ls.remote_checks.is_empty());
    }

    #[test]
    fn host_drops_spoofed_and_far_future_steps() {
        let port = test_port();
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();
        let mut ls = Lockstep::new(vec![Peer::new(server, 1)], true, 1, 2, [false; MAXF], unpack_teams(TEAMS_FFA), 0);

        // The link plays faction 1: a Step claiming faction 0 is spoofed, and
        // a step number far past the lockstep window is garbage.
        send_framed(&mut client, &Msg::Step { faction: 0, step: 5, cmds: vec![Cmd::Cancel { building: 9 }] }).unwrap();
        send_framed(&mut client, &Msg::Step { faction: 1, step: 5, cmds: vec![Cmd::Cancel { building: 9 }] }).unwrap();
        send_framed(&mut client, &Msg::Step { faction: 1, step: 100_000, cmds: Vec::new() }).unwrap();
        for _ in 0..4000 {
            ls.service();
            if ls.cmds[1].contains_key(&5) {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(ls.cmds[1].contains_key(&5));
        assert!(!ls.cmds[0].contains_key(&5), "spoofed faction stored");
        assert!(!ls.cmds[1].contains_key(&100_000), "far-future step stored");
    }
}
