//! Serverless peer-to-peer netcode. The two machines never exchange game state
//! — only commands — because they run the identical deterministic simulation in
//! lockstep. One peer hosts (listens), the other joins (connects). Commands
//! issued now are scheduled a few steps in the future (input delay) and both
//! sides advance a step only once they hold both players' commands for it. A
//! periodic state checksum is swapped to catch any divergence.

use crate::world::{kind_from_u8, Cmd};
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream, UdpSocket};
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
    let mut v = Vec::with_capacity(n);
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
        _ => return None,
    })
}

// ---- messages -------------------------------------------------------------

/// Wire messages. Step/Check carry the originating faction so the host can fan
/// commands out to all peers; Welcome/Start set up the match during the lobby.
enum Msg {
    Step { faction: u8, step: u32, cmds: Vec<Cmd> },
    Check { faction: u8, step: u32, hash: u64 },
    Welcome { faction: u8 },                        // host -> joiner on accept
    Start { seed: u64, factions: u8, ai_mask: u8 }, // host -> joiner to begin
}

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
        Msg::Start { seed, factions, ai_mask } => {
            b.u8(4);
            b.u64(*seed);
            b.u8(*factions);
            b.u8(*ai_mask);
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
        },
        _ => return None,
    })
}

// ---- the transport --------------------------------------------------------

struct Peer {
    stream: TcpStream,
    inbuf: Vec<u8>,
    closed: bool,
}
impl Peer {
    fn send(&mut self, m: &Msg) {
        let bytes = enc_msg(m);
        let len = (bytes.len() as u32).to_le_bytes();
        if self.stream.write_all(&len).is_err() || self.stream.write_all(&bytes).is_err() {
            self.closed = true;
        }
    }
    fn poll(&mut self) -> Vec<Msg> {
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
    let stream = match addr.parse() {
        Ok(sa) => TcpStream::connect_timeout(&sa, Duration::from_secs(6))?,
        Err(_) => TcpStream::connect(addr)?,
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
    pub my_faction: usize,
    pub sim_step: u32,
    sent_step: u32,
    cmds: Vec<HashMap<u32, Vec<Cmd>>>, // [faction][step] -> commands
    local_checks: HashMap<u32, u64>,
    remote_checks: HashMap<u32, u64>,
    pub pending: Vec<Cmd>,
    pub desync_step: Option<u32>,
    pub last_synced: u32,
    pub disconnected: bool,
}

impl Lockstep {
    fn new(peers: Vec<Peer>, is_host: bool, seed: u64, factions: usize, is_ai: [bool; MAXF], my_faction: usize) -> Lockstep {
        Lockstep {
            peers,
            is_host,
            seed,
            factions,
            is_ai,
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
        send_framed(&mut stream, &Msg::Start { seed, factions: 2, ai_mask: 0 })?;
        stream.set_nonblocking(true)?;
        Ok(Lockstep::new(
            vec![Peer { stream, inbuf: Vec::new(), closed: false }],
            true,
            seed,
            2,
            [false; MAXF],
            0,
        ))
    }

    /// Blocking 2-player joiner (faction 1). Used by the headless test.
    pub fn join(addr: &str) -> std::io::Result<Lockstep> {
        let mut stream = connect_to(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(6))).ok();
        let _welcome = read_framed(&mut stream)?;
        let start = read_framed(&mut stream)?;
        let (seed, factions) = match start {
            Msg::Start { seed, factions, .. } => (seed, factions as usize),
            _ => return Err(std::io::Error::new(ErrorKind::InvalidData, "expected start")),
        };
        stream.set_nonblocking(true)?;
        Ok(Lockstep::new(
            vec![Peer { stream, inbuf: Vec::new(), closed: false }],
            false,
            seed,
            factions,
            [false; MAXF],
            1,
        ))
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
        let np = self.peers.len();
        for j in 0..np {
            for m in self.peers[j].poll() {
                match m {
                    Msg::Step { faction, step, cmds } => {
                        let f = faction as usize;
                        if f < self.cmds.len() {
                            self.cmds[f].entry(step).or_insert_with(|| cmds.clone());
                        }
                        if self.is_host {
                            self.relay(j, &Msg::Step { faction, step, cmds });
                        }
                    }
                    Msg::Check { faction, step, hash } => {
                        self.remote_checks.insert(step, hash);
                        if self.is_host {
                            self.relay(j, &Msg::Check { faction, step, hash });
                        }
                    }
                    _ => {} // Welcome/Start are lobby-only
                }
            }
        }
        while self.sent_step < self.sim_step + INPUT_DELAY {
            let cmds = std::mem::take(&mut self.pending);
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

    /// Store a Step/Check that arrived in the same poll batch as the Start
    /// message (so the joiner doesn't drop the host's earliest commands).
    fn ingest(&mut self, m: Msg) {
        match m {
            Msg::Step { faction, step, cmds } => {
                let f = faction as usize;
                if f < self.cmds.len() {
                    self.cmds[f].entry(step).or_insert(cmds);
                }
            }
            Msg::Check { step, hash, .. } => {
                self.remote_checks.insert(step, hash);
            }
            _ => {}
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
        let mut matched: Vec<u32> = Vec::new();
        for (&s, &rh) in &self.remote_checks {
            if let Some(&lh) = self.local_checks.get(&s) {
                if lh == rh {
                    self.last_synced = self.last_synced.max(s);
                    matched.push(s);
                } else {
                    self.desync_step = Some(self.desync_step.map_or(s, |d| d.min(s)));
                }
            }
        }
        for s in matched {
            self.local_checks.remove(&s);
            self.remote_checks.remove(&s);
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
    peers: Vec<Peer>,
    peer_faction: Vec<usize>,
}

impl Host {
    pub fn bind(port: u16, seed: u64, factions: usize) -> std::io::Result<Host> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        listener.set_nonblocking(true)?;
        let beacon = UdpSocket::bind(("0.0.0.0", DISCO_PORT)).ok().and_then(|s| {
            s.set_nonblocking(true).ok()?;
            Some(s)
        });
        Ok(Host { listener, beacon, port, seed, factions: factions.clamp(2, MAXF), peers: Vec::new(), peer_faction: Vec::new() })
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
        // Drop any joiner that has bailed out before the game starts.
        for p in self.peers.iter_mut() {
            let _ = p.poll();
        }
        let alive: Vec<bool> = self.peers.iter().map(|p| !p.closed).collect();
        let mut k = 0;
        self.peers.retain(|_| {
            let keep = alive[k];
            k += 1;
            keep
        });
        // (faction assignments stay with surviving peers by position)
        self.peer_faction = (1..=self.peers.len()).collect();

        if self.peers.len() + 1 < self.factions {
            if let Ok((mut stream, _addr)) = self.listener.accept() {
                let faction = self.peers.len() + 1;
                stream.set_nodelay(true).ok();
                if send_framed(&mut stream, &Msg::Welcome { faction: faction as u8 }).is_ok() && stream.set_nonblocking(true).is_ok() {
                    self.peers.push(Peer { stream, inbuf: Vec::new(), closed: false });
                    self.peer_faction.push(faction);
                }
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
            *slot = f != 0 && !self.peer_faction.contains(&f);
        }
        let ai_mask = (0..MAXF).fold(0u8, |m, f| if is_ai[f] { m | (1 << f) } else { m });
        let start = Msg::Start { seed: self.seed, factions: self.factions as u8, ai_mask };
        for p in &mut self.peers {
            p.send(&start);
        }
        Lockstep::new(self.peers, true, self.seed, self.factions, is_ai, 0)
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
        let my_faction = match read_framed(&mut stream)? {
            Msg::Welcome { faction } => faction as usize,
            _ => return Err(std::io::Error::new(ErrorKind::InvalidData, "expected welcome")),
        };
        stream.set_read_timeout(None).ok();
        stream.set_nonblocking(true)?;
        Ok(Joiner { peer: Some(Peer { stream, inbuf: Vec::new(), closed: false }), my_faction })
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
                Msg::Start { seed, factions, ai_mask } if cfg.is_none() => cfg = Some((seed, factions, ai_mask)),
                other => backlog.push(other),
            }
        }
        let (seed, factions, ai_mask) = cfg?;
        let peer = self.peer.take()?;
        let mut is_ai = [false; MAXF];
        for (f, slot) in is_ai.iter_mut().enumerate() {
            *slot = ai_mask & (1 << f) != 0;
        }
        let mut ls = Lockstep::new(vec![peer], false, seed, factions as usize, is_ai, self.my_faction);
        for m in backlog {
            ls.ingest(m);
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
