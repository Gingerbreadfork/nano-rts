//! The simulation. Entities (bases, barracks, workers, soldiers, minerals)
//! all live in one flat vector and are driven by a fixed-step update: economy,
//! production, target acquisition, combat, movement with local collision
//! avoidance, then death and win/lose resolution. No rendering here — the world
//! only knows how to *be*, not how to look.

use crate::audio::Sfx;
use crate::vec::{v2, V2};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// Base map size (a 2-player match). The actual play area grows with the number
// of factions — see `World::map_size` — so 3- and 4-way games get more room.
pub const WORLD_W: f32 = 2800.0;
pub const WORLD_H: f32 = 1800.0;

// ---- terrain --------------------------------------------------------------
// A coarse tile grid laid over the world. Open ground is the battlefield; high
// ground is a passable plateau with a vision/range edge; cliffs are the
// impassable plateau walls; ramps are the passable way up.
pub const TCELL: f32 = 60.0;
pub const T_OPEN: u8 = 0;
pub const T_HIGH: u8 = 1;
pub const T_CLIFF: u8 = 2;
pub const T_RAMP: u8 = 3;

const CARRY: u32 = 8; // minerals a worker hauls per trip
const MINE_TIME: f32 = 1.3; // seconds to fill up at a patch
const REPAIR_SECONDS: f32 = 14.0; // worker time to fully repair a building from zero
const REPAIR_COST_FRAC: f32 = 0.5; // fraction of the build cost to fully repair
/// Minimum clear gap between a new building and any other building, so clusters
/// stay loose enough for units to path between them.
pub const BUILD_GAP: f32 = 26.0;
pub const MINERAL_START: u32 = 1500;

/// Up to four warring factions plus neutrals. `Player`/`Enemy` are simply
/// factions 0 and 1 (kept by name so the bulk of the 2-player code is unchanged);
/// `Faction2`/`Faction3` extend the model to free-for-alls.
pub const MAX_FACTIONS: usize = 4;
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Team {
    Player,   // faction 0
    Enemy,    // faction 1
    Faction2, // faction 2
    Faction3, // faction 3
    Neutral,
}
impl Team {
    /// Faction index 0..MAX_FACTIONS, or usize::MAX for Neutral.
    #[inline]
    pub fn idx(self) -> usize {
        match self {
            Team::Player => 0,
            Team::Enemy => 1,
            Team::Faction2 => 2,
            Team::Faction3 => 3,
            Team::Neutral => usize::MAX,
        }
    }
    #[inline]
    pub fn from_idx(i: usize) -> Team {
        match i {
            0 => Team::Player,
            1 => Team::Enemy,
            2 => Team::Faction2,
            3 => Team::Faction3,
            _ => Team::Neutral,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Base,     // Command Center: trains workers, drop-off, supplies population
    Barracks, // trains Soldiers and Pyros
    Factory,  // trains Tanks and Raiders
    Depot,    // raises the supply cap
    Worker,
    Soldier, // cheap ranged backbone
    Tank,    // slow, tanky, long-range, splash damage
    Pyro,    // flamethrower: short-range cone of fire that melts clustered units
    Raider,  // fast light vehicle: harass, flanks, runs down workers
    Mortar,  // siege: very long lobbed splash, but a dead zone up close
    Sapper,  // suicide bomber: charges in and detonates for a big blast
    Mineral,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Order {
    Idle,
    Move(V2),
    AttackMove(V2),
    Attack(u32),
    Gather(u32),
    Build(Kind, V2),
    Repair(u32),
}

/// A player action. ALL gameplay input is funnelled through these — in
/// single-player they're applied immediately; in multiplayer they're the only
/// thing sent over the wire (lockstep), and recording them gives replays.
#[derive(Clone, Debug)]
pub enum Cmd {
    Order { ids: Vec<u32>, x: f32, y: f32, attack_move: bool, queue: bool },
    Stop { ids: Vec<u32> },
    Train { building: u32, unit: Kind },
    Build { worker: u32, kind: Kind, x: f32, y: f32, chain: bool },
    Rally { building: u32, x: f32, y: f32 },
    Cancel { building: u32 },
}

pub struct Ent {
    pub id: u32,
    pub team: Team,
    pub kind: Kind,
    pub pos: V2,
    pub hp: f32,
    pub max_hp: f32,
    pub order: Order,
    pub goal: Option<V2>,
    pub cooldown: f32,
    pub carry: u32,
    pub mine_timer: f32,
    pub minerals: u32,        // for mineral patches
    pub queue: Vec<Kind>,     // production queue (bases / barracks)
    pub build_queue: Vec<(Kind, V2)>, // worker's chained build orders (Shift-click)
    pub order_queue: Vec<Order>, // queued waypoint orders (Shift-right-click)
    pub repair_owed: f32,     // fractional minerals a repairing worker still owes
    pub train_timer: f32,
    pub rally: V2,
    pub rally_set: bool,      // player explicitly placed this building's rally
    pub facing: V2,           // unit heading, for drawing rotation
    pub selected: bool,
    pub flash: f32,           // damage flash timer
    pub build_left: f32,      // >0 while a building is still being assembled
    pub path: Vec<V2>,        // remaining pathfinding waypoints toward the goal
    pub repath: f32,          // countdown to the next path recompute
}

pub struct Tracer {
    pub a: V2,
    pub b: V2,
    pub life: f32,
    pub color: u32,
}

pub struct Particle {
    pub pos: V2,
    pub vel: V2,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    pub color: u32,
    pub drag: f32, // velocity damping per second
    pub grav: f32, // downward acceleration
    pub glow: bool, // additive (glowing) vs normal blend
}

/// An expanding shockwave ring from an explosion.
pub struct Shock {
    pub pos: V2,
    pub max_r: f32,
    pub life: f32,
    pub max_life: f32,
    pub color: u32,
}

/// What the opponent is *currently trying to do*. Re-evaluated on an irregular
/// timer and driven by fog-limited intel, so its cadence isn't a predictable
/// wave clock you can set your watch by.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Intent {
    Build,      // macro up; the army gathers at the staging point
    Harass(V2), // peel a small squad off to poke a target (economy / expansion)
    Commit(V2), // throw the whole army at a point
    Feint(V2),  // fake a commit to bait the player out, then peel back
    Defend(V2), // collapse home onto a spotted threat
}

/// A personality rolled at the start of each match so the opponent opens and
/// paces differently every game.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strategy {
    Rush,     // early aggression, worker harass, all-ins
    Macro,    // expand, fat economy, big delayed armies
    Mech,     // rush a factory, tank-heavy, methodical
    Standard, // balanced
}

#[derive(Clone)]
pub struct AiState {
    pub think: f32,
    pub staging: V2,    // where forces gather between attacks
    pub expanded: bool,
    // Rolled per match (see World::roll_ai):
    pub strategy: Strategy,
    pub worker_target: u32, // workers to saturate to (per base)
    pub tank_ratio: f32,    // 0..1 preference for tanks over soldiers
    pub pyro_ratio: f32,    // 0..1 chance a barracks unit is a Pyro
    pub raider_ratio: f32,  // 0..1 chance a factory unit is a Raider
    pub sapper_ratio: f32,  // 0..1 chance a barracks unit is a Sapper
    pub mortar_ratio: f32,  // 0..1 chance a factory unit is a Mortar
    pub expand_min: u32,    // minerals banked before expanding (huge = never)
    pub harass: bool,       // fond of poking the player's economy
    pub aggression: f32,    // 0..1: commits earlier, with less of an edge
    pub patience: f32,      // ~how long it holds an intent before re-deciding
    // Live intent — the irregular cadence that replaces the old wave clock:
    pub intent: Intent,
    pub intent_timer: f32, // re-evaluate the intent when this reaches 0
    pub commit_army: u32,  // army supply at the moment the current attack began
    // Fog-limited intel — what the opponent actually *knows*, not omniscience:
    pub scout_id: u32,       // unit currently scouting (0 = none)
    pub scout_timer: f32,    // cooldown before dispatching the next scout
    pub player_main: V2,     // the player's start base (fixed-spawn geography)
    pub seen_army_pos: V2,   // last place the player's army was spotted
    pub seen_army_supply: u32, // strength of the player army last spotted
    pub seen_army_age: f32,  // seconds since the army was actually in sight
    pub known: Vec<(V2, Kind)>, // player buildings we've scouted (raid targets)
}
impl AiState {
    fn fresh() -> AiState {
        AiState {
            think: 0.0,
            staging: v2(0.0, 0.0),
            expanded: false,
            strategy: Strategy::Standard,
            worker_target: 12,
            tank_ratio: 0.3,
            pyro_ratio: 0.25,
            raider_ratio: 0.3,
            sapper_ratio: 0.12,
            mortar_ratio: 0.18,
            expand_min: 500,
            harass: false,
            aggression: 0.5,
            patience: 5.0,
            intent: Intent::Build,
            intent_timer: 0.0,
            commit_army: 0,
            scout_id: 0,
            scout_timer: 6.0,
            player_main: v2(0.0, 0.0),
            seen_army_pos: v2(0.0, 0.0),
            seen_army_supply: 0,
            seen_army_age: 99.0,
            known: Vec::new(),
        }
    }
}

pub struct World {
    pub ents: Vec<Ent>,
    next_id: u32,
    /// Minerals per faction, indexed by `Team::idx()`.
    pub minerals: [u32; MAX_FACTIONS],
    /// Number of active factions (2..=4).
    pub factions: usize,
    /// Which factions are AI-controlled (the rest are humans).
    pub is_ai: [bool; MAX_FACTIONS],
    /// Map dimensions (scale with the faction count).
    pub world_w: f32,
    pub world_h: f32,
    /// Which team the local player controls/views. Player for single-player and
    /// the host; another faction for each joining peer in multiplayer.
    pub my_team: Team,
    /// True for a networked match (affects UI; AI fill is driven by `is_ai`).
    pub versus: bool,
    pub cam: V2,
    pub view_w: f32, // local viewport size, set each frame — gates screen-wide
    pub view_h: f32, // explosion effects (flash/shake) to what's on or near screen
    pub tracers: Vec<Tracer>,
    pub particles: Vec<Particle>,
    pub shocks: Vec<Shock>,
    pub flash_amt: f32,  // full-screen flash intensity, decays
    pub flash_color: u32,
    pub shake: f32,      // screen-shake intensity, decays
    pub shake_off: V2,   // current shake offset (applied at render)
    // Sound events for this tick (drained by the audio engine). A position
    // tags world sounds for distance/pan attenuation; None = a global UI cue.
    pub sounds: Vec<(Sfx, Option<V2>)>,
    last_shot_snd: f32,
    // Fog of war, PER FACTION: `factions` grids laid end-to-end, each cell
    // 0 = unseen, 1 = explored, 2 = visible. The local player views their own
    // grid; each AI only reacts to what falls inside its own grid (not
    // omniscient). Computed for every active faction so AI fog stays
    // deterministic across peers.
    pub vis: Vec<u8>,
    pub fog_w: usize,
    pub fog_h: usize,
    pub fog_cell: f32,
    // Terrain tile grid (T_OPEN/T_HIGH/T_CLIFF/T_RAMP), generated from the seed.
    pub terrain: Vec<u8>,
    pub tw: usize,
    pub th: usize,
    pub has_cliffs: bool, // any cliffs at all (gates the cliff hard-stop)
    // Dynamic building-occupancy grid (1 = a building sits here), rebuilt each
    // tick so pathfinding routes AROUND buildings, not into them.
    pub block: Vec<u8>,
    pub pings: Vec<(V2, f32, u32)>, // command feedback: pos, life, color
    pub messages: Vec<(String, f32)>,
    rng: u64,
    // A SEPARATE RNG for cosmetics (particles, screen shake). Kept off `rng` —
    // which is hashed into the desync checksum — so visual code can never
    // perturb the lockstep simulation. Not checksummed; may differ per peer.
    fx_rng: u64,
    pub over: i32,         // local result: 0 running, 1 victory, -1 defeat
    pub match_over: bool,  // global: <=1 faction left — the sim freezes for all
    pub time: f32,
    pub ai: Vec<AiState>, // one AI brain per faction (only AI factions act)
    pub attack_warn: f32,
    pub kills: u32,
}

// ---- per-kind stats -------------------------------------------------------

pub fn radius(k: Kind) -> f32 {
    match k {
        Kind::Base => 30.0,
        Kind::Barracks => 22.0,
        Kind::Factory => 24.0,
        Kind::Depot => 16.0,
        Kind::Worker => 7.0,
        Kind::Soldier => 8.0,
        Kind::Tank => 11.0,
        Kind::Pyro => 8.0,
        Kind::Raider => 9.0,
        Kind::Mortar => 11.0,
        Kind::Sapper => 8.0,
        Kind::Mineral => 14.0,
    }
}
/// Fog-of-war sight radius (world units). Minerals reveal nothing.
fn sight(k: Kind) -> f32 {
    match k {
        Kind::Base => 300.0,
        Kind::Barracks | Kind::Factory | Kind::Depot => 210.0,
        Kind::Worker => 175.0,
        Kind::Soldier => 215.0,
        Kind::Tank => 245.0,
        Kind::Pyro => 190.0,
        Kind::Raider => 240.0, // fast eyes — a natural scout
        Kind::Mortar => 235.0, // sees far to lob far
        Kind::Sapper => 185.0,
        Kind::Mineral => 0.0,
    }
}
fn speed(k: Kind) -> f32 {
    match k {
        Kind::Worker => 94.0,
        Kind::Soldier => 82.0,
        Kind::Tank => 52.0,
        Kind::Pyro => 74.0,
        Kind::Raider => 138.0, // by far the fastest thing on the field
        Kind::Mortar => 46.0,  // slowest — it has to set up
        Kind::Sapper => 116.0, // sprints into the fight
        _ => 0.0,
    }
}
fn max_hp(k: Kind) -> f32 {
    match k {
        Kind::Base => 2000.0,
        Kind::Barracks => 700.0,
        Kind::Factory => 850.0,
        Kind::Depot => 400.0,
        Kind::Worker => 40.0,
        Kind::Soldier => 55.0,
        Kind::Tank => 240.0,
        Kind::Pyro => 110.0, // soaks a bit so it can close the distance
        Kind::Raider => 70.0,
        Kind::Mortar => 85.0, // a glass siege weapon
        Kind::Sapper => 45.0, // fragile; it only needs to reach you once
        Kind::Mineral => 1.0,
    }
}
fn damage(k: Kind) -> f32 {
    match k {
        Kind::Worker => 3.0,
        Kind::Soldier => 9.0,
        Kind::Tank => 24.0,
        Kind::Pyro => 9.0, // per tick, but hits everything in the cone
        Kind::Raider => 16.0,
        Kind::Mortar => 30.0, // a heavy shell, lands as splash
        Kind::Sapper => 0.0,  // no gun — all its damage is the detonation
        _ => 0.0,
    }
}
/// Reach *beyond* the two bodies' radii.
fn atk_range(k: Kind) -> f32 {
    match k {
        Kind::Worker => 3.0,
        Kind::Soldier => 80.0,
        Kind::Tank => 110.0,
        Kind::Pyro => 42.0, // must get right up close
        Kind::Raider => 56.0,
        Kind::Mortar => 215.0, // outranges everything
        Kind::Sapper => 14.0,  // the range at which it touches off
        _ => 0.0,
    }
}
/// Mortars can't fire at anything closer than this — a dead zone that makes them
/// helpless if you get on top of them. Zero for everything else.
fn min_range(k: Kind) -> f32 {
    match k {
        Kind::Mortar => 85.0,
        _ => 0.0,
    }
}
/// Radius of a Sapper's suicide blast (and the damage at its centre).
fn sapper_blast() -> (f32, f32) {
    (58.0, 85.0)
}
fn atk_cd(k: Kind) -> f32 {
    match k {
        Kind::Worker => 0.7,
        Kind::Soldier => 0.85,
        Kind::Tank => 1.5,
        Kind::Pyro => 0.4, // a rapid wash of fire
        Kind::Raider => 0.8,
        Kind::Mortar => 2.4, // a slow reload between shells
        Kind::Sapper => 0.1,
        _ => 0.0,
    }
}
/// Auto-acquire radius. Workers don't go looking for fights.
fn aggro(k: Kind) -> f32 {
    match k {
        Kind::Soldier => 165.0,
        Kind::Tank => 200.0,
        Kind::Pyro => 130.0,
        Kind::Raider => 175.0,
        Kind::Mortar => 240.0, // engages from way out
        Kind::Sapper => 205.0, // charges the nearest threat
        _ => 0.0,
    }
}
/// Tanks deal their damage to everything within this radius of the impact.
fn splash(k: Kind) -> f32 {
    match k {
        Kind::Tank => 28.0,
        Kind::Mortar => 42.0, // a wide shell burst
        _ => 0.0,
    }
}
/// Flamethrower cone: `(reach, cos of the half-angle)`. None for non-flamers.
/// The Pyro damages every enemy inside this arc in front of it each tick.
fn flame(k: Kind) -> Option<(f32, f32)> {
    match k {
        Kind::Pyro => Some((78.0, 0.80)), // ~37 degree half-angle
        _ => None,
    }
}
pub fn cost(k: Kind) -> u32 {
    match k {
        Kind::Worker => 50,
        Kind::Soldier => 50,
        Kind::Tank => 150,
        Kind::Pyro => 70,
        Kind::Raider => 65,
        Kind::Mortar => 145,
        Kind::Sapper => 55,
        Kind::Barracks => 120,
        Kind::Factory => 160,
        Kind::Depot => 75,
        Kind::Base => 300, // expansion
        _ => 0,
    }
}
pub fn build_time(k: Kind) -> f32 {
    match k {
        Kind::Worker => 5.0,
        Kind::Soldier => 6.0,
        Kind::Tank => 9.0,
        Kind::Pyro => 7.0,
        Kind::Raider => 6.0,
        Kind::Mortar => 11.0,
        Kind::Sapper => 5.0,
        Kind::Barracks => 10.0,
        Kind::Factory => 13.0,
        Kind::Depot => 6.0,
        Kind::Base => 28.0,
        _ => 0.0,
    }
}
/// Population a unit consumes.
pub fn supply_cost(k: Kind) -> u32 {
    match k {
        Kind::Worker | Kind::Soldier => 1,
        Kind::Pyro | Kind::Raider | Kind::Sapper => 2,
        Kind::Tank | Kind::Mortar => 3,
        _ => 0,
    }
}
/// Population a building provides.
pub fn supply_provide(k: Kind) -> u32 {
    match k {
        Kind::Base => 11,
        Kind::Depot => 8,
        _ => 0,
    }
}

/// Stable byte for a Kind (for command serialization over the wire).
pub fn kind_from_u8(b: u8) -> Kind {
    match b {
        0 => Kind::Base,
        1 => Kind::Barracks,
        2 => Kind::Factory,
        3 => Kind::Depot,
        4 => Kind::Worker,
        5 => Kind::Soldier,
        6 => Kind::Tank,
        7 => Kind::Pyro,
        8 => Kind::Raider,
        9 => Kind::Mortar,
        10 => Kind::Sapper,
        _ => Kind::Mineral,
    }
}

fn is_mover(k: Kind) -> bool {
    matches!(
        k,
        Kind::Worker | Kind::Soldier | Kind::Tank | Kind::Pyro | Kind::Raider | Kind::Mortar | Kind::Sapper
    )
}
pub fn is_building(k: Kind) -> bool {
    matches!(k, Kind::Base | Kind::Barracks | Kind::Factory | Kind::Depot)
}
fn is_combat(k: Kind) -> bool {
    // Things that fire a straight tracer (drives the tracer/muzzle visuals). The
    // Sapper detonates and the Mortar lobs an arcing shell, so both get their own
    // bespoke effects instead.
    matches!(k, Kind::Soldier | Kind::Tank | Kind::Pyro | Kind::Raider)
}
/// An army unit (everything that fights, excludes workers) — for Ctrl+A and HUD.
pub fn is_army(k: Kind) -> bool {
    matches!(
        k,
        Kind::Soldier | Kind::Tank | Kind::Pyro | Kind::Raider | Kind::Mortar | Kind::Sapper
    )
}

/// A faction-neutral tie-break key for "nearest target" scans. Entity ids run in
/// spawn order, so faction 0 (the human) always holds the lowest ids; comparing
/// raw ids on a distance tie would always point at it. Hashing scrambles that
/// correlation so ties fall fairly across factions. Deterministic.
pub fn tie_key(id: u32) -> u64 {
    (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Scramble a raw seed (splitmix64) into a well-mixed, odd RNG state. Without
/// this, xorshift64's first outputs from small/low-entropy seeds are tiny and
/// correlated — which biased the personality roll for sequential seeds.
fn mix_seed(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

/// A hashable fingerprint of an order (for the desync checksum).
fn order_disc(o: &Order) -> u64 {
    match o {
        Order::Idle => 1,
        Order::Move(p) => 2 ^ (p.x.to_bits() as u64) ^ ((p.y.to_bits() as u64) << 1),
        Order::AttackMove(p) => 3 ^ (p.x.to_bits() as u64) ^ ((p.y.to_bits() as u64) << 1),
        Order::Attack(id) => 4 ^ (*id as u64),
        Order::Gather(id) => 5 ^ (*id as u64),
        Order::Build(k, p) => 6 ^ (*k as u64) ^ (p.x.to_bits() as u64),
        Order::Repair(id) => 7 ^ (*id as u64),
    }
}

impl World {
    /// Classic 2-player game: a human (Player) versus one AI (Enemy).
    pub fn new(seed: u64) -> World {
        World::new_match(seed, 2, [false, true, false, false], Team::Player, false)
    }

    /// The map size for `factions` players (grows with the count).
    pub fn map_size(factions: usize) -> (f32, f32) {
        let s = 1.0 + 0.18 * (factions.max(2) - 2) as f32;
        (WORLD_W * s, WORLD_H * s)
    }

    /// Build a match with `factions` (2..=4) factions; `is_ai[i]` marks AI slots;
    /// `my_team` is the local viewer. `versus` flags a networked game.
    pub fn new_match(seed: u64, factions: usize, is_ai: [bool; MAX_FACTIONS], my_team: Team, versus: bool) -> World {
        let factions = factions.clamp(2, MAX_FACTIONS);
        let (world_w, world_h) = World::map_size(factions);
        let fog_cell = 40.0f32;
        let fog_w = (world_w / fog_cell).ceil() as usize;
        let fog_h = (world_h / fog_cell).ceil() as usize;
        let tw = (world_w / TCELL).ceil() as usize;
        let th = (world_h / TCELL).ceil() as usize;
        let mut w = World {
            ents: Vec::new(),
            next_id: 1,
            minerals: [50; MAX_FACTIONS],
            factions,
            is_ai,
            world_w,
            world_h,
            my_team,
            versus,
            cam: v2(0.0, 0.0),
            view_w: 1280.0,
            view_h: 720.0,
            tracers: Vec::new(),
            particles: Vec::new(),
            shocks: Vec::new(),
            flash_amt: 0.0,
            flash_color: 0xFFFF_FFFF,
            shake: 0.0,
            shake_off: v2(0.0, 0.0),
            sounds: Vec::new(),
            last_shot_snd: -1.0,
            vis: vec![0u8; fog_w * fog_h * factions],
            fog_w,
            fog_h,
            fog_cell,
            terrain: vec![T_OPEN; tw * th],
            tw,
            th,
            has_cliffs: false,
            block: vec![0u8; tw * th],
            pings: Vec::new(),
            messages: Vec::new(),
            rng: mix_seed(seed),
            fx_rng: mix_seed(seed ^ 0x1234_5678_9ABC_DEF0),
            over: 0,
            match_over: false,
            time: 0.0,
            ai: (0..factions).map(|_| AiState::fresh()).collect(),
            attack_warn: 0.0,
            kills: 0,
        };
        w.setup();
        w
    }

    fn setup(&mut self) {
        let (w, h) = (self.world_w, self.world_h);
        let (mx, my) = (w * 0.13, h * 0.18);
        // Corner start slots; the first `factions` are used. 2p is diagonal.
        let corners = [
            v2(mx, h - my),     // 0: bottom-left
            v2(w - mx, my),     // 1: top-right
            v2(mx, my),         // 2: top-left
            v2(w - mx, h - my), // 3: bottom-right
        ];
        let center = v2(w * 0.5, h * 0.5);
        let n = self.factions;
        let mut base_pos = [v2(0.0, 0.0); MAX_FACTIONS];

        // Which faction starts in which corner. With 3+ players the corners are
        // not all equal (bottom slots fare worse), so locking the human to slot 0
        // would hand it the weakest spot every game. Shuffle the assignment from
        // the seed instead — fair for the human, identical on every peer. A 2p
        // duel is already symmetric (diagonal starts), so leave it untouched.
        let mut slot = [0usize, 1, 2, 3];
        if n >= 3 {
            for i in (1..4).rev() {
                let j = (self.rng_u() % (i as u64 + 1)) as usize;
                slot.swap(i, j);
            }
        }

        for fi in 0..n {
            let t = Team::from_idx(fi);
            let bp = corners[slot[fi]];
            base_pos[fi] = bp;
            let fwd = center.sub(bp).norm(); // toward the middle
            let perp = v2(-fwd.y, fwd.x);
            let bi = self.spawn(Kind::Base, t, bp);
            self.ents[bi].rally = bp.add(fwd.scale(130.0));
            // Mineral line between the base and the map centre.
            self.mineral_field(bp.add(fwd.scale(160.0)), 7);
            // Four workers, already mining, fanned out toward the line.
            for i in 0..4 {
                let wp = bp.add(fwd.scale(78.0)).add(perp.scale(i as f32 * 20.0 - 30.0));
                let id = self.spawn(Kind::Worker, t, wp);
                self.auto_gather(id);
            }
        }

        // Contested patches in the middle, more of them on bigger maps.
        self.mineral_field(center, 6);
        if n >= 3 {
            self.mineral_field(v2(w * 0.5, h * 0.22), 5);
            self.mineral_field(v2(w * 0.5, h * 0.78), 5);
        }
        if n >= 4 {
            self.mineral_field(v2(w * 0.22, h * 0.5), 5);
            self.mineral_field(v2(w * 0.78, h * 0.5), 5);
        }

        // Carve the battlefield (bases/minerals kept clear, all bases connected).
        self.gen_terrain();

        // Configure each AI faction: roll a personality, set a forward staging
        // point and its nearest enemy base as the opening objective.
        for fi in 0..n {
            if self.is_ai[fi] {
                self.roll_ai(fi);
                let bp = base_pos[fi];
                let fwd = center.sub(bp).norm();
                self.ai[fi].staging = bp.add(fwd.scale(380.0));
                let mut best = center;
                let mut bd = f32::MAX;
                for (fj, &op) in base_pos.iter().enumerate().take(n) {
                    if fj != fi && op.dist_sq(bp) < bd {
                        bd = op.dist_sq(bp);
                        best = op;
                    }
                }
                self.ai[fi].player_main = best;
                self.ai[fi].seen_army_pos = best;
            }
        }

        // Centre the camera on the local player's start.
        let mybase = base_pos[self.my_team.idx().min(n - 1)];
        self.cam = mybase.sub(v2(640.0, 360.0));
        self.clamp_cam(1280.0, 720.0);
        let foes = n - 1;
        self.msg(&format!("FREE-FOR-ALL - {} RIVAL{} - LAST ONE STANDING WINS", foes, if foes == 1 { "" } else { "S" }));
    }

    fn mineral_field(&mut self, center: V2, n: i32) {
        for i in 0..n {
            let ang = (i as f32 / n as f32) * std::f32::consts::TAU;
            let p = center.add(v2(ang.cos() * 70.0, ang.sin() * 46.0));
            let idx = self.spawn(Kind::Mineral, Team::Neutral, p);
            self.ents[idx].minerals = MINERAL_START;
        }
    }

    fn rng_u(&mut self) -> u64 {
        // xorshift64
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }
    /// Uniform random float in 0..1 (exposed so the AI can vary its play).
    pub fn rng_f(&mut self) -> f32 {
        (self.rng_u() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Cosmetic-only random float in 0..1, drawn from `fx_rng` so particle and
    /// shake jitter never touches the checksummed simulation RNG.
    fn fx_f(&mut self) -> f32 {
        let mut x = self.fx_rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.fx_rng = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Roll a personality + jittered parameters for AI faction `fi`, so each
    /// opponent opens and paces differently every game.
    fn roll_ai(&mut self, fi: usize) {
        // Pick from the float (high, well-mixed bits): xorshift's low bits are
        // weak, so `rng_u() % 4` would bias the strategy by the seed's parity.
        let strat = match (self.rng_f() * 4.0) as u32 {
            0 => Strategy::Rush,
            1 => Strategy::Macro,
            2 => Strategy::Mech,
            _ => Strategy::Standard,
        };
        let (a, b, c, d, e) = (
            self.rng_f(),
            self.rng_f(),
            self.rng_f(),
            self.rng_f(),
            self.rng_f(),
        );
        // (worker_target, tank_ratio, expand_min, harass, aggression, patience)
        let (wt, tr, em, har, agg, pat) = match strat {
            // Rush: few workers, soldier-heavy, never expands, loves to harass,
            // very aggressive and twitchy (short patience).
            Strategy::Rush => (8 + (a * 3.0) as u32, 0.1 + b * 0.2, 900, true, 0.75 + c * 0.2, 2.5 + d * 1.5),
            // Macro: fat economy, big delayed armies, expands early, patient.
            Strategy::Macro => (14 + (a * 5.0) as u32, 0.35 + b * 0.25, 320, e < 0.4, 0.25 + c * 0.25, 6.0 + d * 3.0),
            // Mech: tank-heavy, methodical, middling aggression.
            Strategy::Mech => (11 + (a * 3.0) as u32, 0.55 + b * 0.3, 600, e < 0.25, 0.4 + c * 0.25, 5.0 + d * 2.5),
            // Standard: balanced, with the widest spread of temperaments.
            Strategy::Standard => (11 + (a * 4.0) as u32, 0.3 + b * 0.2, 500, e < 0.35, 0.35 + c * 0.4, 3.5 + d * 3.0),
        };
        // Specialist mix, rolled fresh so two same-strategy opponents still feel
        // different: how much it leans on flame (anti-clump) and raiders (harass).
        let pyro = (self.rng_f() * 0.4).min(0.4);
        let raider = match strat {
            Strategy::Rush => 0.25 + self.rng_f() * 0.35, // rushers love fast raiders
            _ => self.rng_f() * 0.35,
        };
        // Sappers (anti-clump suicide) and mortars (siege) round out the mix —
        // rushers favour sappers, mech/macro favour siege.
        let sapper = match strat {
            Strategy::Rush => 0.1 + self.rng_f() * 0.25,
            _ => self.rng_f() * 0.18,
        };
        let mortar = match strat {
            Strategy::Mech | Strategy::Macro => 0.12 + self.rng_f() * 0.25,
            _ => self.rng_f() * 0.18,
        };
        let ai = &mut self.ai[fi];
        ai.strategy = strat;
        ai.worker_target = wt;
        ai.tank_ratio = tr;
        ai.expand_min = em;
        ai.harass = har;
        ai.aggression = agg;
        ai.patience = pat;
        ai.pyro_ratio = pyro;
        ai.raider_ratio = raider;
        ai.sapper_ratio = sapper;
        ai.mortar_ratio = mortar;
    }

    pub fn spawn(&mut self, kind: Kind, team: Team, pos: V2) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let mh = max_hp(kind);
        self.ents.push(Ent {
            id,
            team,
            kind,
            pos,
            hp: mh,
            max_hp: mh,
            order: Order::Idle,
            goal: None,
            cooldown: 0.0,
            carry: 0,
            mine_timer: 0.0,
            minerals: 0,
            queue: Vec::new(),
            build_queue: Vec::new(),
            order_queue: Vec::new(),
            repair_owed: 0.0,
            train_timer: 0.0,
            rally: pos,
            rally_set: false,
            facing: v2(0.0, 1.0),
            selected: false,
            flash: 0.0,
            build_left: 0.0,
            path: Vec::new(),
            repath: 0.0,
        });
        self.ents.len() - 1
    }

    pub fn index_of(&self, id: u32) -> Option<usize> {
        self.ents.iter().position(|e| e.id == id)
    }

    /// First living base for a team. Used by the AI brain.
    pub fn first_base(&self, team: Team) -> Option<usize> {
        self.ents
            .iter()
            .position(|e| e.kind == Kind::Base && e.team == team && e.hp > 0.0)
    }

    /// Nearest mineral patch — exposed so the AI can route its workers.
    pub fn nearest_mineral_idx(&self, p: V2) -> Option<usize> {
        self.nearest_mineral(p)
    }

    /// Position of the nearest enemy within `r` of `pt`, if any. Lets minimap
    /// attack-clicks snap onto a target without pixel-perfect aim.
    pub fn snap_to_enemy(&self, pt: V2, r: f32) -> Option<V2> {
        let mut best = None;
        let mut bd = r * r;
        for e in &self.ents {
            if e.team != Team::Enemy || e.hp <= 0.0 {
                continue;
            }
            let d = e.pos.dist_sq(pt);
            if d < bd {
                bd = d;
                best = Some(e.pos);
            }
        }
        best
    }

    /// Is a building footprint clear (in-bounds and not overlapping solids)?
    pub fn can_build(&self, kind: Kind, pt: V2) -> bool {
        let r = radius(kind);
        if pt.x < r || pt.y < r || pt.x > self.world_w - r || pt.y > self.world_h - r {
            return false;
        }
        // No building on cliffs or astride a ramp (don't wall off the high ground).
        if self.has_cliffs {
            for &(ox, oy) in &[(0.0, 0.0), (-r, 0.0), (r, 0.0), (0.0, -r), (0.0, r)] {
                let t = self.tile_at(v2(pt.x + ox, pt.y + oy));
                if t == T_CLIFF || t == T_RAMP {
                    return false;
                }
            }
        }
        for e in &self.ents {
            if is_mover(e.kind) {
                continue;
            }
            // Keep a clear gap between buildings so they don't box units in;
            // minerals only need a hair of clearance (bases hug their patch).
            let gap = if is_building(e.kind) { BUILD_GAP } else { 8.0 };
            if e.pos.dist(pt) < r + radius(e.kind) + gap {
                return false;
            }
        }
        // Also stay clear of buildings ordered but not yet raised — a worker's
        // active build site and any chained ones it has queued — so chained
        // placements (and the AI's own pending builds) never overlap.
        for e in &self.ents {
            if e.kind != Kind::Worker {
                continue;
            }
            if let Order::Build(bk, site) = e.order {
                if site.dist(pt) < r + radius(bk) + BUILD_GAP {
                    return false;
                }
            }
            for &(bk, site) in &e.build_queue {
                if site.dist(pt) < r + radius(bk) + BUILD_GAP {
                    return false;
                }
            }
        }
        true
    }

    /// Building placement: spend, validate spacing, send the worker. Replaces any
    /// in-progress build chain (a plain, single build).
    pub fn order_build(&mut self, worker_idx: usize, kind: Kind, site: V2) -> bool {
        let team = self.ents[worker_idx].team;
        if self.team_min(team) < cost(kind) || !self.can_build(kind, site) {
            return false;
        }
        self.spend(team, cost(kind));
        self.ents[worker_idx].build_queue.clear();
        self.ents[worker_idx].order = Order::Build(kind, site);
        true
    }

    /// Chained build (Shift-click): reserve the cost now and append to the
    /// worker's build queue, so it raises each one in turn. Starts immediately if
    /// the worker isn't already building.
    pub fn queue_build(&mut self, worker_idx: usize, kind: Kind, site: V2) -> bool {
        let team = self.ents[worker_idx].team;
        if self.team_min(team) < cost(kind) || !self.can_build(kind, site) {
            return false;
        }
        self.spend(team, cost(kind));
        let busy = matches!(self.ents[worker_idx].order, Order::Build(_, _))
            || !self.ents[worker_idx].build_queue.is_empty();
        if busy {
            self.ents[worker_idx].build_queue.push((kind, site));
        } else {
            self.ents[worker_idx].order = Order::Build(kind, site);
        }
        true
    }

    /// Drop a worker's whole build plan — the one it's walking to plus every
    /// chained one — and refund their cost (nothing was raised yet). Used when a
    /// worker is deliberately sent off to do something else.
    fn cancel_builds(&mut self, i: usize) {
        let team = self.ents[i].team;
        let mut refund = 0u32;
        if let Order::Build(k, _) = self.ents[i].order {
            refund += cost(k);
        }
        for &(k, _) in &self.ents[i].build_queue {
            refund += cost(k);
        }
        self.ents[i].build_queue.clear();
        if refund > 0 {
            self.gain(team, refund);
        }
    }

    // ---- queries ----------------------------------------------------------

    fn nearest_enemy(&self, team: Team, p: V2, within: f32) -> Option<usize> {
        let mut best = None;
        let mut bd = within * within;
        let mut bk = 0u64;
        for (i, e) in self.ents.iter().enumerate() {
            if e.team == Team::Neutral || e.team == team || e.hp <= 0.0 {
                continue;
            }
            let d = e.pos.dist_sq(p);
            if d > bd {
                continue;
            }
            // Break exact ties (common with symmetric starts) by a hash of the
            // entity id, NOT by list order — otherwise the faction spawned first
            // (always the human, slot 0) gets targeted on every tie and quietly
            // soaks extra focus fire.
            let k = tie_key(e.id);
            if best.is_none() || d < bd || k < bk {
                bd = d;
                bk = k;
                best = Some(i);
            }
        }
        best
    }

    fn nearest_mineral(&self, p: V2) -> Option<usize> {
        let mut best = None;
        let mut bd = f32::MAX;
        for (i, e) in self.ents.iter().enumerate() {
            if e.kind != Kind::Mineral || e.minerals == 0 {
                continue;
            }
            let d = e.pos.dist_sq(p);
            if d < bd {
                bd = d;
                best = Some(i);
            }
        }
        best
    }

    /// Choose a patch for a worker near `p`: nearest, but biased away from
    /// patches other workers already mine, so a fresh crew fans out across the
    /// field instead of dog-piling the closest clump. `exclude` is the asking
    /// worker's own id (so it doesn't count itself). Deterministic: scans ents
    /// in order, integer-free tie-break by first-lowest score.
    fn pick_patch(&self, p: V2, exclude: u32) -> Option<usize> {
        // One worker already on a patch makes it count as ~170px farther — enough
        // to prefer an empty neighbour in the same line, never enough to send a
        // worker off to a different field across the map.
        const CROWD: f32 = 30_000.0;
        let mut best = None;
        let mut bd = f32::MAX;
        for (i, e) in self.ents.iter().enumerate() {
            if e.kind != Kind::Mineral || e.minerals == 0 {
                continue;
            }
            let pid = e.id;
            let assigned = self
                .ents
                .iter()
                .filter(|w| {
                    w.id != exclude
                        && w.kind == Kind::Worker
                        && matches!(w.order, Order::Gather(g) if g == pid)
                })
                .count() as f32;
            let score = e.pos.dist_sq(p) + assigned * CROWD;
            if score < bd {
                bd = score;
                best = Some(i);
            }
        }
        best
    }

    fn nearest_base(&self, team: Team, p: V2) -> Option<usize> {
        let mut best = None;
        let mut bd = f32::MAX;
        for (i, e) in self.ents.iter().enumerate() {
            if e.kind != Kind::Base || e.team != team {
                continue;
            }
            let d = e.pos.dist_sq(p);
            if d < bd {
                bd = d;
                best = Some(i);
            }
        }
        best
    }

    fn pick(&self, world_pt: V2) -> Option<usize> {
        // Topmost entity under a point (units win over buildings on ties).
        let mut best = None;
        let mut bd = f32::MAX;
        for (i, e) in self.ents.iter().enumerate() {
            let r = radius(e.kind) + 3.0;
            let d = e.pos.dist_sq(world_pt);
            if d <= r * r && d < bd {
                bd = d;
                best = Some(i);
            }
        }
        best
    }

    // ---- economy / production --------------------------------------------

    pub fn team_min(&self, t: Team) -> u32 {
        self.minerals.get(t.idx()).copied().unwrap_or(0)
    }
    fn spend(&mut self, t: Team, amt: u32) {
        if let Some(m) = self.minerals.get_mut(t.idx()) {
            *m = m.saturating_sub(amt);
        }
    }
    fn gain(&mut self, t: Team, amt: u32) {
        if let Some(m) = self.minerals.get_mut(t.idx()) {
            *m += amt;
        }
    }

    /// Population currently consumed by a team's units.
    pub fn supply_used(&self, team: Team) -> u32 {
        self.ents
            .iter()
            .filter(|e| e.team == team)
            .map(|e| supply_cost(e.kind))
            .sum()
    }
    /// Population a team's buildings provide (its army ceiling).
    pub fn supply_cap(&self, team: Team) -> u32 {
        let cap: u32 = self
            .ents
            .iter()
            .filter(|e| e.team == team && e.build_left <= 0.0)
            .map(|e| supply_provide(e.kind))
            .sum();
        cap.min(120)
    }

    /// Queue a unit at a building if affordable (minerals + supply). Returns success.
    pub fn try_train(&mut self, bidx: usize, kind: Kind) -> bool {
        let team = self.ents[bidx].team;
        let c = cost(kind);
        if self.team_min(team) < c {
            if team == self.my_team {
                self.msg("NOT ENOUGH MINERALS");
            }
            return false;
        }
        // Count already-queued units against the cap so we don't overproduce.
        let queued: u32 = self
            .ents
            .iter()
            .filter(|e| e.team == team)
            .flat_map(|e| e.queue.iter())
            .map(|&q| supply_cost(q))
            .sum();
        if self.supply_used(team) + queued + supply_cost(kind) > self.supply_cap(team) {
            if team == self.my_team {
                self.msg("NEED A SUPPLY DEPOT");
            }
            return false;
        }
        if self.ents[bidx].build_left > 0.0 {
            return false; // still under construction
        }
        self.spend(team, c);
        self.ents[bidx].queue.push(kind);
        true
    }

    /// Of the player's selected, ready production buildings of `bkind`, the id of
    /// the one with the shortest queue (ties to the lowest id). Drives even
    /// distribution: with several barracks selected, each train press lands on
    /// whichever is least loaded, so they fill in lock-step instead of piling
    /// onto the first. Returns None if none are selected/ready.
    pub fn least_loaded_selected(&self, bkind: Kind) -> Option<u32> {
        self.ents
            .iter()
            .filter(|e| {
                e.selected && e.team == self.my_team && e.kind == bkind && e.build_left <= 0.0
            })
            .min_by_key(|e| (e.queue.len(), e.id))
            .map(|e| e.id)
    }

    fn auto_gather(&mut self, idx: usize) {
        let wid = self.ents[idx].id;
        if let Some(m) = self.pick_patch(self.ents[idx].pos, wid) {
            let id = self.ents[m].id;
            self.ents[idx].order = Order::Gather(id);
        }
    }

    // ---- player commands --------------------------------------------------

    pub fn clear_selection(&mut self) {
        for e in self.ents.iter_mut() {
            e.selected = false;
        }
    }

    pub fn select_single(&mut self, world_pt: V2, additive: bool) {
        if !additive {
            self.clear_selection();
        }
        // Forgiving, player-preferring pick: grab the nearest friendly body
        // within a generous click tolerance so small units aren't fiddly.
        let mut best = None;
        let mut bd = f32::MAX;
        for (i, e) in self.ents.iter().enumerate() {
            if e.team != self.my_team {
                continue;
            }
            let tol = radius(e.kind) + 12.0;
            let d = e.pos.dist_sq(world_pt);
            if d <= tol * tol && d < bd {
                bd = d;
                best = Some(i);
            }
        }
        if let Some(i) = best {
            self.ents[i].selected = true;
        }
    }

    /// Double-click behaviour. On a unit: select every player unit of that type
    /// currently on screen. On a building: select every same-kind building of
    /// ours nearby (a production cluster), so they can be rallied or queued as a
    /// group with even distribution.
    pub fn select_type_in_view(&mut self, world_pt: V2, vmin: V2, vmax: V2, additive: bool) {
        let mt = self.my_team;
        // What did the click land on? Nearest friendly body, any kind.
        let hit = {
            let mut found: Option<(Kind, V2)> = None;
            let mut bd = f32::MAX;
            for e in self.ents.iter() {
                if e.team != mt {
                    continue;
                }
                let tol = radius(e.kind) + 12.0;
                let d = e.pos.dist_sq(world_pt);
                if d <= tol * tol && d < bd {
                    bd = d;
                    found = Some((e.kind, e.pos));
                }
            }
            found
        };
        match hit {
            Some((kind, at)) if is_building(kind) => {
                if !additive {
                    self.clear_selection();
                }
                let r2 = 540.0 * 540.0;
                for e in self.ents.iter_mut() {
                    if e.team == mt && e.kind == kind && e.pos.dist_sq(at) <= r2 {
                        e.selected = true;
                    }
                }
            }
            Some((kind, _)) => {
                if !additive {
                    self.clear_selection();
                }
                for e in self.ents.iter_mut() {
                    if e.team == mt
                        && e.kind == kind
                        && e.pos.x >= vmin.x
                        && e.pos.x <= vmax.x
                        && e.pos.y >= vmin.y
                        && e.pos.y <= vmax.y
                    {
                        e.selected = true;
                    }
                }
            }
            None => self.select_single(world_pt, additive),
        }
    }

    /// Select every combat unit the player owns across the whole map — workers
    /// and buildings excluded. Bound to Ctrl+A: "grab the army".
    pub fn select_all_army(&mut self, additive: bool) {
        if !additive {
            self.clear_selection();
        }
        let mt = self.my_team;
        for e in self.ents.iter_mut() {
            if e.team == mt && is_army(e.kind) {
                e.selected = true;
            }
        }
    }

    /// IDs of the current player selection — for storing a control group.
    pub fn selected_ids(&self) -> Vec<u32> {
        let mt = self.my_team;
        self.ents
            .iter()
            .filter(|e| e.selected && e.team == mt)
            .map(|e| e.id)
            .collect()
    }

    /// Restore a selection from stored IDs, skipping any that have died.
    pub fn select_ids(&mut self, ids: &[u32], additive: bool) {
        if !additive {
            self.clear_selection();
        }
        let mt = self.my_team;
        for e in self.ents.iter_mut() {
            if e.team == mt && ids.contains(&e.id) {
                e.selected = true;
            }
        }
    }

    pub fn centroid_of_ids(&self, ids: &[u32]) -> Option<V2> {
        let mut sum = v2(0.0, 0.0);
        let mut n = 0;
        for e in &self.ents {
            if ids.contains(&e.id) {
                sum = sum.add(e.pos);
                n += 1;
            }
        }
        if n == 0 {
            None
        } else {
            Some(sum.scale(1.0 / n as f32))
        }
    }

    pub fn select_box(&mut self, a: V2, b: V2, additive: bool) {
        let (x0, x1) = (a.x.min(b.x), a.x.max(b.x));
        let (y0, y1) = (a.y.min(b.y), a.y.max(b.y));
        if !additive {
            self.clear_selection();
        }
        let mut any_unit = false;
        let mt = self.my_team;
        for e in self.ents.iter_mut() {
            if e.team == mt
                && is_mover(e.kind)
                && e.pos.x >= x0
                && e.pos.x <= x1
                && e.pos.y >= y0
                && e.pos.y <= y1
            {
                e.selected = true;
                any_unit = true;
            }
        }
        // Tiny box and nothing caught: treat as a click on whatever's there.
        if !any_unit && a.dist(b) < 6.0 {
            self.select_single(a, additive);
        }
    }

    /// (workers, army units, has_base, has_barracks, has_factory) in the
    /// current selection — drives the contextual HUD hints.
    pub fn selected_kinds(&self) -> (u32, u32, bool, bool, bool, bool) {
        let mut w = 0;
        let mut army = 0;
        let mut base = false;
        let mut barr = false;
        let mut fact = false;
        let mut depot = false;
        for e in &self.ents {
            if !e.selected {
                continue;
            }
            match e.kind {
                Kind::Worker => w += 1,
                k if is_army(k) => army += 1,
                Kind::Base => base = true,
                Kind::Barracks => barr = true,
                Kind::Factory => fact = true,
                Kind::Depot => depot = true,
                _ => {}
            }
        }
        (w, army, base, barr, fact, depot)
    }

    /// Apply a serialized command for `team`. The one path all gameplay input
    /// flows through (single-player, replay, and the lockstep network alike).
    pub fn apply_cmd(&mut self, team: Team, cmd: &Cmd) {
        match cmd {
            Cmd::Order { ids, x, y, attack_move, queue } => self.apply_order(team, ids, v2(*x, *y), *attack_move, *queue),
            Cmd::Stop { ids } => self.apply_stop(team, ids),
            Cmd::Train { building, unit } => {
                if let Some(i) = self.index_of(*building) {
                    if self.ents[i].team == team && is_building(self.ents[i].kind) {
                        self.try_train(i, *unit);
                    }
                }
            }
            Cmd::Build { worker, kind, x, y, chain: _ } => {
                // Always append onto the worker's build chain (it starts straight
                // away if idle). A placement never clears an existing queue, so you
                // can finish placing, click away, or deselect and the worker still
                // raises everything it was told to.
                if let Some(i) = self.index_of(*worker) {
                    if self.ents[i].team == team && self.ents[i].kind == Kind::Worker {
                        self.queue_build(i, *kind, v2(*x, *y));
                    }
                }
            }
            Cmd::Rally { building, x, y } => {
                if let Some(i) = self.index_of(*building) {
                    if self.ents[i].team == team && is_building(self.ents[i].kind) {
                        self.ents[i].rally = v2(*x, *y);
                        self.ents[i].rally_set = true;
                    }
                }
            }
            Cmd::Cancel { building } => {
                if let Some(i) = self.index_of(*building) {
                    if self.ents[i].team == team && is_building(self.ents[i].kind) && !self.ents[i].queue.is_empty() {
                        let k = self.ents[i].queue.pop().unwrap();
                        if self.ents[i].queue.is_empty() {
                            self.ents[i].train_timer = 0.0;
                        }
                        self.gain(team, cost(k));
                    }
                }
            }
        }
    }

    /// Right-click order for the local player's selection (convenience wrapper,
    /// used by the test harness; live play routes through `apply_cmd`).
    #[allow(dead_code)]
    pub fn command(&mut self, world_pt: V2, attack_move: bool) {
        let ids: Vec<u32> = self
            .ents
            .iter()
            .filter(|e| e.selected && e.team == self.my_team)
            .map(|e| e.id)
            .collect();
        self.apply_order(self.my_team, &ids, world_pt, attack_move, false);
    }

    /// Issue a move/attack/gather/rally/cancel to specific units owned by `team`,
    /// resolving the action from whatever is under `world_pt`. Team-generic.
    pub fn apply_order(&mut self, team: Team, ids: &[u32], world_pt: V2, attack_move: bool, queue: bool) {
        let target = self.pick(world_pt);
        // A move onto a cliff snaps to the nearest reachable ground, so a misclick
        // never strands units pushing into a wall.
        let world_pt = self.snap_open(world_pt);
        let mine = team == self.my_team; // visual/audio feedback only for the local player

        // Right-clicking your own selected producing building cancels a queued unit.
        if let Some(ti) = target {
            if self.ents[ti].team == team
                && ids.contains(&self.ents[ti].id)
                && is_building(self.ents[ti].kind)
                && !self.ents[ti].queue.is_empty()
            {
                let kind = self.ents[ti].queue.pop().unwrap();
                if self.ents[ti].queue.is_empty() {
                    self.ents[ti].train_timer = 0.0;
                }
                self.gain(team, cost(kind));
                if mine {
                    let p = self.ents[ti].pos;
                    self.pings.push((p, 0.5, crate::gfx::rgb(240, 120, 90)));
                    self.sfx(Sfx::Select);
                    self.msg("CANCELLED");
                }
                return;
            }
        }

        let tinfo = target.map(|i| (self.ents[i].id, self.ents[i].team, self.ents[i].kind));
        // A friendly, damaged building the click landed on — a repair target.
        let repair_id = target
            .filter(|&ti| {
                self.ents[ti].team == team
                    && is_building(self.ents[ti].kind)
                    && self.ents[ti].hp < self.ents[ti].max_hp
            })
            .map(|ti| self.ents[ti].id);
        let sel: Vec<usize> = ids
            .iter()
            .filter_map(|&id| self.index_of(id))
            .filter(|&i| self.ents[i].team == team)
            .collect();
        let any_worker = sel.iter().any(|&i| self.ents[i].kind == Kind::Worker);

        if mine && sel.iter().any(|&i| is_mover(self.ents[i].kind)) {
            let (pt, col) = match tinfo {
                Some((_, t, _)) if t != team && t != Team::Neutral => {
                    (self.ents[target.unwrap()].pos, crate::gfx::rgb(240, 90, 80))
                }
                Some((_, _, Kind::Mineral)) => {
                    (self.ents[target.unwrap()].pos, crate::gfx::rgb(90, 210, 220))
                }
                _ if repair_id.is_some() && any_worker => {
                    (self.ents[target.unwrap()].pos, crate::gfx::rgb(120, 230, 200))
                }
                _ if attack_move => (world_pt, crate::gfx::rgb(255, 200, 90)),
                _ => (world_pt, crate::gfx::rgb(120, 230, 140)),
            };
            self.pings.push((pt, 0.5, col));
        }

        let movers: Vec<usize> = sel.iter().cloned().filter(|&i| is_mover(self.ents[i].kind)).collect();
        let count = movers.len().max(1);
        let cols = (count as f32).sqrt().ceil() as i32;
        let mut placed = 0i32;
        for &i in &sel {
            let kind = self.ents[i].kind;
            if is_building(kind) {
                // Only buildings that train units have a meaningful rally point;
                // a Supply Depot produces nothing, so ignore the click.
                if matches!(kind, Kind::Base | Kind::Barracks | Kind::Factory) {
                    self.ents[i].rally = world_pt;
                    self.ents[i].rally_set = true;
                    if mine {
                        self.pings.push((world_pt, 0.6, crate::gfx::rgb(120, 220, 140)));
                    }
                }
                continue;
            }
            // Decide the order this unit should carry out.
            let new_order = match tinfo {
                Some((tid, tteam, _)) if tteam != team && tteam != Team::Neutral => Order::Attack(tid),
                Some((tid, _, Kind::Mineral)) if kind == Kind::Worker => Order::Gather(tid),
                _ if kind == Kind::Worker && repair_id.is_some() => Order::Repair(repair_id.unwrap()),
                _ => {
                    let c = placed % cols;
                    let r = placed / cols;
                    let off = v2((c - cols / 2) as f32 * 22.0, (r - cols / 2) as f32 * 22.0);
                    let dst = world_pt.add(off);
                    placed += 1;
                    if attack_move { Order::AttackMove(dst) } else { Order::Move(dst) }
                }
            };
            // A worker being sent off mid-build is a deliberate redirect: drop and
            // refund its build chain rather than leave it orphaned, and take the
            // new order straight away (don't queue it behind the build).
            let redirect_builder = kind == Kind::Worker
                && (matches!(self.ents[i].order, Order::Build(_, _)) || !self.ents[i].build_queue.is_empty());
            if redirect_builder {
                self.cancel_builds(i);
                self.ents[i].order_queue.clear();
                self.ents[i].order = new_order;
            } else if queue && !matches!(self.ents[i].order, Order::Idle) {
                // Shift-queue appends a waypoint; a plain order replaces the queue.
                self.ents[i].order_queue.push(new_order);
            } else {
                self.ents[i].order_queue.clear();
                self.ents[i].order = new_order;
            }
        }
    }

    pub fn apply_stop(&mut self, team: Team, ids: &[u32]) {
        for &id in ids {
            if let Some(i) = self.index_of(id) {
                if self.ents[i].team == team && is_mover(self.ents[i].kind) {
                    if self.ents[i].kind == Kind::Worker {
                        self.cancel_builds(i); // halt and refund any pending build chain
                    }
                    self.ents[i].order = Order::Idle;
                    self.ents[i].goal = None;
                    self.ents[i].order_queue.clear();
                }
            }
        }
    }

    /// A 64-bit fingerprint of the simulation state, exchanged in multiplayer to
    /// detect any divergence (desync) between the two peers.
    pub fn checksum(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(self.next_id as u64);
        for fi in 0..self.factions {
            mix(self.minerals[fi] as u64);
        }
        mix(self.time.to_bits() as u64);
        mix(self.rng);
        for e in &self.ents {
            mix(e.id as u64);
            mix(e.kind as u64);
            mix(e.team as u64);
            mix(e.pos.x.to_bits() as u64);
            mix(e.pos.y.to_bits() as u64);
            mix(e.hp.to_bits() as u64);
            mix(e.carry as u64);
            mix(e.minerals as u64);
            mix(e.queue.len() as u64);
            mix(e.build_queue.len() as u64);
            mix(e.order_queue.len() as u64);
            mix(order_disc(&e.order));
        }
        h
    }

    pub fn msg(&mut self, s: &str) {
        self.messages.push((s.to_string(), 3.2));
        if self.messages.len() > 5 {
            self.messages.remove(0);
        }
    }

    /// Queue a sound for the audio engine (bounded so a headless sim can't grow it).
    /// A global cue (UI clicks, the under-attack alarm) — always audible.
    fn sfx(&mut self, s: Sfx) {
        self.push_sound(s, None);
    }
    /// A world sound at `pos` — the audio engine fades it by distance from view.
    fn sfx_at(&mut self, s: Sfx, pos: V2) {
        self.push_sound(s, Some(pos));
    }
    fn push_sound(&mut self, s: Sfx, pos: Option<V2>) {
        self.sounds.push((s, pos));
        if self.sounds.len() > 64 {
            self.sounds.remove(0);
        }
    }

    // ---- particle effects -------------------------------------------------

    /// Spray `count` particles outward from `pos`. `glow` draws them additively.
    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, pos: V2, color: u32, count: u32, speed: f32, size: f32, life: f32, grav: f32, drag: f32, glow: bool) {
        for _ in 0..count {
            let a = self.fx_f() * std::f32::consts::TAU;
            let s = speed * (0.25 + 0.75 * self.fx_f());
            let l = life * (0.55 + 0.45 * self.fx_f());
            self.particles.push(Particle {
                pos,
                vel: v2(a.cos() * s, a.sin() * s),
                life: l,
                max_life: l,
                size,
                color,
                drag,
                grav,
                glow,
            });
        }
        let n = self.particles.len();
        if n > 2400 {
            self.particles.drain(0..(n - 2400));
        }
    }

    /// Spray particles in a cone around `dir` (`spread` = half-angle, radians),
    /// with a little positional jitter. The directional cousin of `emit` — used
    /// for the flamethrower's gout of fire.
    #[allow(clippy::too_many_arguments)]
    fn emit_cone(&mut self, pos: V2, dir: V2, spread: f32, color: u32, count: u32, speed: f32, size: f32, life: f32, grav: f32, drag: f32, glow: bool) {
        let base = dir.y.atan2(dir.x);
        for _ in 0..count {
            let a = base + (self.fx_f() - 0.5) * 2.0 * spread;
            let s = speed * (0.45 + 0.55 * self.fx_f());
            let l = life * (0.6 + 0.5 * self.fx_f());
            let jx = (self.fx_f() - 0.5) * 5.0;
            let jy = (self.fx_f() - 0.5) * 5.0;
            self.particles.push(Particle {
                pos: pos.add(v2(jx, jy)),
                vel: v2(a.cos() * s, a.sin() * s),
                life: l,
                max_life: l,
                size,
                color,
                drag,
                grav,
                glow,
            });
        }
        let n = self.particles.len();
        if n > 2400 {
            self.particles.drain(0..(n - 2400));
        }
    }

    /// The flamethrower's tongue of fire: layered cones running white-hot at the
    /// nozzle, through yellow/orange flame, to smoky red tips — plus the odd
    /// arcing ember. Additive glow makes dense fire blow out to white.
    fn flame_fx(&mut self, nozzle: V2, dir: V2, reach: f32) {
        let white = crate::gfx::rgb(255, 245, 205);
        let yellow = crate::gfx::rgb(255, 208, 90);
        let orange = crate::gfx::rgb(255, 125, 38);
        let red = crate::gfx::rgb(206, 52, 28);
        let smoke = crate::gfx::rgb(62, 56, 58);
        let spread = 0.40;
        let v = reach * 5.0;
        self.emit_cone(nozzle, dir, spread * 0.55, white, 3, v * 0.9, 2.2, 0.10, 0.0, 7.5, true);
        self.emit_cone(nozzle, dir, spread, yellow, 4, v * 0.95, 2.7, 0.16, -12.0, 5.0, true);
        self.emit_cone(nozzle, dir, spread, orange, 5, v * 0.85, 3.1, 0.24, -20.0, 4.2, true);
        self.emit_cone(nozzle, dir, spread * 1.2, red, 3, v * 0.7, 2.7, 0.34, -28.0, 3.4, true);
        self.emit_cone(nozzle, dir, spread * 1.35, smoke, 2, v * 0.5, 3.6, 0.55, -42.0, 2.4, false);
        if self.fx_f() < 0.4 {
            self.emit_cone(nozzle, dir, spread, yellow, 1, v * 1.15, 1.6, 0.5, 130.0, 1.5, true);
        }
    }

    fn add_shake(&mut self, amt: f32) {
        self.shake = (self.shake + amt).min(18.0);
    }
    /// 0..1 strength an explosion at `pos` should register on the LOCAL view:
    /// full inside the viewport, fading out a screen away. Camera-relative and
    /// cosmetic — it never touches the checksummed sim — so a flash/shake never
    /// fires for a blast clear across the map you can't even see.
    fn view_gain(&self, pos: V2) -> f32 {
        let dx = (pos.x - (self.cam.x + self.view_w * 0.5)).abs() - self.view_w * 0.5;
        let dy = (pos.y - (self.cam.y + self.view_h * 0.5)).abs() - self.view_h * 0.5;
        let outside = dx.max(0.0).max(dy.max(0.0));
        (1.0 - outside / 420.0).clamp(0.0, 1.0)
    }
    fn shock(&mut self, pos: V2, max_r: f32, life: f32, color: u32) {
        self.shocks.push(Shock { pos, max_r, life, max_life: life, color });
        if self.shocks.len() > 64 {
            self.shocks.remove(0);
        }
    }
    fn flash(&mut self, amt: f32, color: u32) {
        if amt > self.flash_amt {
            self.flash_amt = amt;
            self.flash_color = color;
        }
    }

    /// Death explosion — sized, coloured, and dramatised by what just died.
    fn death_fx(&mut self, pos: V2, kind: Kind, team: Team) {
        let white = crate::gfx::rgb(255, 255, 240);
        let hot = crate::gfx::rgb(255, 230, 130);
        let warm = crate::gfx::rgb(255, 140, 50);
        let smoke = crate::gfx::rgb(80, 80, 88);
        let tcol = match team {
            Team::Player => crate::gfx::rgb(120, 180, 255),
            Team::Enemy => crate::gfx::rgb(255, 110, 95),
            _ => crate::gfx::rgb(160, 160, 170),
        };
        // Screen-wide flash/shake only fire for blasts on or near the view.
        let g = self.view_gain(pos);
        match kind {
            Kind::Tank => {
                self.emit(pos, white, 10, 90.0, 3.5, 0.18, 0.0, 8.0, true); // core flash
                self.emit(pos, hot, 24, 260.0, 3.0, 0.5, 120.0, 4.0, true);
                self.emit(pos, warm, 30, 190.0, 2.5, 0.7, 70.0, 3.0, true);
                self.emit(pos, smoke, 16, 60.0, 5.0, 1.2, -55.0, 1.5, false);
                self.shock(pos, 70.0, 0.4, hot);
                self.flash(0.35 * g, warm);
                self.add_shake(6.0 * g);
                self.sfx_at(Sfx::BigBoom, pos);
            }
            Kind::Soldier => {
                self.emit(pos, white, 4, 60.0, 2.2, 0.14, 0.0, 9.0, true);
                self.emit(pos, hot, 14, 170.0, 2.2, 0.4, 90.0, 4.0, true);
                self.emit(pos, tcol, 10, 120.0, 2.0, 0.4, 90.0, 4.0, true);
                self.shock(pos, 26.0, 0.25, tcol);
                self.add_shake(0.8 * g);
                self.sfx_at(Sfx::Explosion, pos);
            }
            Kind::Worker => {
                self.emit(pos, tcol, 10, 120.0, 2.0, 0.35, 90.0, 4.0, true);
                self.emit(pos, hot, 5, 110.0, 1.8, 0.3, 90.0, 5.0, true);
                self.sfx_at(Sfx::Explosion, pos);
            }
            Kind::Pyro => {
                // The fuel tank cooks off — a hot, fiery burst that lingers.
                self.emit(pos, white, 6, 80.0, 2.6, 0.16, 0.0, 8.0, true);
                self.emit(pos, hot, 22, 200.0, 2.8, 0.55, 30.0, 3.5, true);
                self.emit(pos, warm, 26, 150.0, 3.0, 0.8, -30.0, 2.8, true); // rising flame
                self.emit(pos, smoke, 12, 60.0, 4.2, 1.1, -45.0, 1.6, false);
                self.shock(pos, 40.0, 0.3, warm);
                self.flash(0.22 * g, warm);
                self.add_shake(1.6 * g);
                self.sfx_at(Sfx::Explosion, pos);
            }
            Kind::Raider => {
                // A light vehicle going up — sharper than a soldier, smaller than a tank.
                self.emit(pos, white, 5, 80.0, 2.4, 0.15, 0.0, 8.0, true);
                self.emit(pos, hot, 16, 200.0, 2.6, 0.45, 110.0, 4.0, true);
                self.emit(pos, tcol, 12, 150.0, 2.2, 0.45, 110.0, 4.0, true);
                self.emit(pos, smoke, 6, 70.0, 3.6, 0.9, -40.0, 1.8, false);
                self.shock(pos, 30.0, 0.26, hot);
                self.add_shake(1.4 * g);
                self.sfx_at(Sfx::Explosion, pos);
            }
            Kind::Mortar => {
                // Stowed shells cook off — a fierce secondary blast.
                self.emit(pos, white, 8, 90.0, 2.8, 0.16, 0.0, 7.0, true);
                self.emit(pos, hot, 24, 230.0, 3.0, 0.55, 60.0, 3.5, true);
                self.emit(pos, warm, 20, 170.0, 2.8, 0.7, 30.0, 3.0, true);
                self.emit(pos, smoke, 10, 60.0, 4.5, 1.1, -40.0, 1.6, false);
                self.shock(pos, 48.0, 0.34, hot);
                self.flash(0.22 * g, warm);
                self.add_shake(3.0 * g);
                self.sfx_at(Sfx::BigBoom, pos);
            }
            Kind::Sapper => {
                // The whole point of the unit — a big, fiery detonation.
                self.emit(pos, white, 12, 120.0, 3.4, 0.18, 0.0, 6.0, true);
                self.emit(pos, hot, 34, 300.0, 3.6, 0.6, 50.0, 3.0, true);
                self.emit(pos, warm, 30, 220.0, 3.2, 0.85, 20.0, 2.6, true);
                self.emit(pos, smoke, 16, 70.0, 5.5, 1.3, -35.0, 1.2, false);
                self.shock(pos, sapper_blast().0, 0.45, hot);
                self.shock(pos, sapper_blast().0 * 0.6, 0.34, white);
                self.flash(0.4 * g, warm);
                self.add_shake(6.0 * g);
                self.sfx_at(Sfx::BigBoom, pos);
            }
            Kind::Mineral => {
                self.emit(pos, crate::gfx::rgb(160, 245, 255), 12, 110.0, 2.2, 0.6, 50.0, 3.0, true);
                self.shock(pos, 22.0, 0.3, crate::gfx::rgb(120, 230, 245));
            }
            _ => {
                // a building — the big one
                self.emit(pos, white, 18, 130.0, 4.5, 0.22, 0.0, 6.0, true); // blinding core
                self.emit(pos, hot, 40, 320.0, 4.0, 0.7, 160.0, 3.0, true);
                self.emit(pos, warm, 48, 230.0, 3.5, 0.9, 90.0, 2.5, true);
                self.emit(pos, smoke, 28, 80.0, 6.5, 1.6, -35.0, 1.0, false);
                self.emit(pos, tcol, 30, 200.0, 3.0, 0.8, 130.0, 3.0, true);
                let big = kind == Kind::Base;
                self.shock(pos, if big { 200.0 } else { 150.0 }, 0.6, hot);
                self.shock(pos, if big { 130.0 } else { 95.0 }, 0.45, white);
                self.flash(if big { 0.8 * g } else { 0.6 * g }, warm);
                self.add_shake(if big { 14.0 * g } else { 9.0 * g });
                self.sfx_at(Sfx::BigBoom, pos);
            }
        }
    }

    fn update_particles(&mut self, dt: f32) {
        for p in self.particles.iter_mut() {
            let d = (1.0 - p.drag * dt).max(0.0);
            p.vel = p.vel.scale(d);
            p.vel.y += p.grav * dt;
            p.pos = p.pos.add(p.vel.scale(dt));
            p.life -= dt;
        }
        self.particles.retain(|p| p.life > 0.0);
        for s in self.shocks.iter_mut() {
            s.life -= dt;
        }
        self.shocks.retain(|s| s.life > 0.0);
        self.flash_amt = (self.flash_amt - dt * 3.0).max(0.0);
        if self.shake > 0.05 {
            self.shake = (self.shake - dt * 22.0).max(0.0);
            let a = self.fx_f() * std::f32::consts::TAU;
            self.shake_off = v2(a.cos(), a.sin()).scale(self.shake);
        } else {
            self.shake = 0.0;
            self.shake_off = v2(0.0, 0.0);
        }
    }

    // ---- terrain ----------------------------------------------------------

    /// Tile code at a world point (off-map reads as an impassable wall).
    #[inline]
    pub fn tile_at(&self, p: V2) -> u8 {
        let x = (p.x / TCELL) as i32;
        let y = (p.y / TCELL) as i32;
        if x < 0 || y < 0 || x as usize >= self.tw || y as usize >= self.th {
            return T_CLIFF;
        }
        self.terrain[y as usize * self.tw + x as usize]
    }

    /// Is this point inside an impassable cliff?
    #[inline]
    pub fn blocked_pt(&self, p: V2) -> bool {
        self.tile_at(p) == T_CLIFF
    }

    /// Elevation tier at a point: 1 on high ground, 0 elsewhere. Drives the
    /// high-ground vision/range edge.
    #[inline]
    pub fn elev_at(&self, p: V2) -> u8 {
        if self.tile_at(p) == T_HIGH {
            1
        } else {
            0
        }
    }

    /// A tile blocked for *pathfinding* — a cliff OR a building sits on it.
    /// (Movement still slides only off cliffs; buildings are handled by
    /// path-routing plus the soft collision push.)
    #[inline]
    fn point_solid(&self, p: V2) -> bool {
        let x = (p.x / TCELL) as i32;
        let y = (p.y / TCELL) as i32;
        if x < 0 || y < 0 || x as usize >= self.tw || y as usize >= self.th {
            return true;
        }
        let i = y as usize * self.tw + x as usize;
        self.terrain[i] == T_CLIFF || self.block[i] != 0
    }

    /// Does the straight segment a->b cross a cliff or a building? Cheap test
    /// for whether a unit needs to path around an obstacle at all.
    fn line_blocked(&self, a: V2, b: V2) -> bool {
        let d = b.sub(a);
        let len = d.len();
        if len < 1.0 {
            return false;
        }
        let steps = (len / (TCELL * 0.5)).ceil() as i32;
        for i in 1..=steps {
            if self.point_solid(a.add(d.scale(i as f32 / steps as f32))) {
                return true;
            }
        }
        false
    }

    /// Like `line_blocked` but only for *cliffs* (impassable terrain), ignoring
    /// the soft building/mineral grid. A unit may steer straight onto a soft
    /// obstacle near its goal (the patch it mines, its own base), but it must
    /// always route around a hard ridge — even one only a short hop away.
    fn line_blocked_cliff(&self, a: V2, b: V2) -> bool {
        let d = b.sub(a);
        let len = d.len();
        if len < 1.0 {
            return false;
        }
        let steps = (len / (TCELL * 0.5)).ceil() as i32;
        for i in 1..=steps {
            if self.blocked_pt(a.add(d.scale(i as f32 / steps as f32))) {
                return true;
            }
        }
        false
    }

    fn cell_passable(&self, cx: i32, cy: i32) -> bool {
        cx >= 0
            && cy >= 0
            && (cx as usize) < self.tw
            && (cy as usize) < self.th
            && self.terrain[cy as usize * self.tw + cx as usize] != T_CLIFF
            && self.block[cy as usize * self.tw + cx as usize] == 0
    }

    /// Rebuild the occupancy grid so A* routes around buildings and mineral
    /// patches without walling off the lanes between them. We mark a tile only
    /// when the obstacle covers its *centre* (plus the tile it stands in, so it is
    /// never invisible). A centred building blocks just its own 60px tile; one
    /// straddling an edge blocks both halves it covers. Mineral patches are
    /// included so units path around a field instead of wedging into a deposit —
    /// the steer-direct-when-close rule still lets workers reach the patch they
    /// are actually mining. Deterministic (ents in order).
    fn rebuild_block_grid(&mut self) {
        for b in self.block.iter_mut() {
            *b = 0;
        }
        let cs = TCELL;
        let (tw, th) = (self.tw as i32, self.th as i32);
        let footprints: Vec<(V2, f32)> = self
            .ents
            .iter()
            .filter(|e| (is_building(e.kind) && e.hp > 0.0) || (e.kind == Kind::Mineral && e.minerals > 0))
            .map(|e| (e.pos, radius(e.kind)))
            .collect();
        let mut mark = |tx: i32, ty: i32| {
            if tx >= 0 && tx < tw && ty >= 0 && ty < th {
                self.block[ty as usize * self.tw + tx as usize] = 1;
            }
        };
        for (c, rad) in footprints {
            let ccx = (c.x / cs) as i32;
            let ccy = (c.y / cs) as i32;
            mark(ccx, ccy); // the tile it stands in, always
            let r2 = rad * rad;
            let rc = (rad / cs).ceil() as i32 + 1;
            for ty in (ccy - rc)..=(ccy + rc) {
                for tx in (ccx - rc)..=(ccx + rc) {
                    // Tile centre inside the building disc.
                    let tcx = (tx as f32 + 0.5) * cs;
                    let tcy = (ty as f32 + 0.5) * cs;
                    if (tcx - c.x) * (tcx - c.x) + (tcy - c.y) * (tcy - c.y) <= r2 {
                        mark(tx, ty);
                    }
                }
            }
        }
    }

    /// Nearest passable cell to a (possibly blocked) cell, by spiral search.
    fn nearest_passable(&self, cx: i32, cy: i32) -> Option<(i32, i32)> {
        if self.cell_passable(cx, cy) {
            return Some((cx, cy));
        }
        for r in 1..(self.tw.max(self.th) as i32) {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue; // ring only
                    }
                    if self.cell_passable(cx + dx, cy + dy) {
                        return Some((cx + dx, cy + dy));
                    }
                }
            }
        }
        None
    }

    /// A* over the tile grid (8-connected, no corner-cutting). Returns smoothed
    /// world-space waypoints from `from` to `to`, or None if unreachable.
    /// Deterministic: integer costs, insertion-ordered tie-breaks.
    pub fn find_path(&self, from: V2, to: V2) -> Option<Vec<V2>> {
        let tw = self.tw as i32;
        let th = self.th as i32;
        let cell = |p: V2| ((p.x / TCELL) as i32, (p.y / TCELL) as i32);
        let (sx0, sy0) = cell(from);
        let (gx0, gy0) = cell(to);
        let (sx, sy) = self.nearest_passable(sx0, sy0)?;
        let (gx, gy) = self.nearest_passable(gx0, gy0)?;
        if (sx, sy) == (gx, gy) {
            return Some(vec![to]);
        }
        let n = (tw * th) as usize;
        let heur = |x: i32, y: i32| -> i32 {
            let dx = (x - gx).abs();
            let dy = (y - gy).abs();
            10 * (dx + dy) + (14 - 2 * 10) * dx.min(dy) // octile
        };
        let mut gcost = vec![i32::MAX; n];
        let mut came = vec![-1i32; n];
        let sidx = (sy * tw + sx) as usize;
        let gidx = (gy * tw + gx) as usize;
        gcost[sidx] = 0;
        let mut heap: BinaryHeap<Reverse<(i32, i32, i32)>> = BinaryHeap::new();
        let mut counter = 0i32;
        heap.push(Reverse((heur(sx, sy), 0, sidx as i32)));
        const NB: [(i32, i32, i32); 8] = [
            (1, 0, 10), (-1, 0, 10), (0, 1, 10), (0, -1, 10),
            (1, 1, 14), (1, -1, 14), (-1, 1, 14), (-1, -1, 14),
        ];
        while let Some(Reverse((_, _, ci))) = heap.pop() {
            let ci = ci as usize;
            if ci == gidx {
                break;
            }
            let cx = ci as i32 % tw;
            let cy = ci as i32 / tw;
            let gc = gcost[ci];
            for &(dx, dy, cost) in &NB {
                let nx = cx + dx;
                let ny = cy + dy;
                if !self.cell_passable(nx, ny) {
                    continue;
                }
                if dx != 0 && dy != 0 && (!self.cell_passable(cx + dx, cy) || !self.cell_passable(cx, cy + dy)) {
                    continue; // don't cut across a cliff corner
                }
                let ni = (ny * tw + nx) as usize;
                let ng = gc + cost;
                if ng < gcost[ni] {
                    gcost[ni] = ng;
                    came[ni] = ci as i32;
                    counter += 1;
                    heap.push(Reverse((ng + heur(nx, ny), counter, ni as i32)));
                }
            }
        }
        if gcost[gidx] == i32::MAX {
            return None;
        }
        // Reconstruct cell path, then string-pull it into long straight runs.
        let mut cells = Vec::new();
        let mut cur = gidx as i32;
        while cur >= 0 {
            cells.push(cur);
            cur = came[cur as usize];
        }
        cells.reverse();
        let mut wps: Vec<V2> = cells
            .iter()
            .skip(1)
            .map(|&ci| {
                let cx = ci % tw;
                let cy = ci / tw;
                v2((cx as f32 + 0.5) * TCELL, (cy as f32 + 0.5) * TCELL)
            })
            .collect();
        if let Some(last) = wps.last_mut() {
            *last = to; // walk to the real goal, not the cell centre
        }
        Some(self.smooth(from, wps))
    }

    /// Drop waypoints we can reach past in a straight line (funnel smoothing).
    fn smooth(&self, start: V2, wps: Vec<V2>) -> Vec<V2> {
        if wps.is_empty() {
            return wps;
        }
        let mut out = Vec::new();
        let mut anchor = start;
        let mut i = 0;
        while i < wps.len() {
            let mut j = i;
            while j + 1 < wps.len() && !self.line_blocked(anchor, wps[j + 1]) {
                j += 1;
            }
            out.push(wps[j]);
            anchor = wps[j];
            i = j + 1;
        }
        out
    }

    /// Generate the map's terrain from the RNG: a couple of point-symmetric
    /// plateaus (high ground ringed by cliffs, reached by ramps), with the
    /// area around bases and minerals kept clear and base-to-base connectivity
    /// guaranteed.
    fn gen_terrain(&mut self) {
        let (tw, th) = (self.tw, self.th);
        let (twi, thi) = (tw as i32, th as i32);
        // Roll plateau specs up front so we're not borrowing rng mid-paint.
        let count = 2 + if self.rng_f() < 0.5 { 1 } else { 0 };
        let mut specs: Vec<(i32, i32, i32, i32, i32)> = Vec::new();
        for _ in 0..count {
            let cx = (8.0 + self.rng_f() * 16.0) as i32; // central band, left half
            let cy = (5.0 + self.rng_f() * 14.0) as i32;
            let hw = (2.0 + self.rng_f() * 1.5) as i32;
            let hh = (2.0 + self.rng_f() * 1.0) as i32;
            let side = (self.rng_f() * 4.0) as i32;
            specs.push((cx, cy, hw, hh, side));
        }

        let mut t = vec![T_OPEN; tw * th];
        let idx = |x: i32, y: i32| y as usize * tw + x as usize;
        let inb = |x: i32, y: i32| x >= 0 && y >= 0 && x < twi && y < thi;

        // High-ground rectangles, each placed with its 180°-rotated mirror.
        for &(cx, cy, hw, hh, _) in &specs {
            for &(mx, my) in &[(cx, cy), (twi - 1 - cx, thi - 1 - cy)] {
                for y in (my - hh)..=(my + hh) {
                    for x in (mx - hw)..=(mx + hw) {
                        if inb(x, y) {
                            t[idx(x, y)] = T_HIGH;
                        }
                    }
                }
            }
        }
        // Cliff ring: any open cell orthogonally touching high ground.
        let snap = t.clone();
        for y in 0..thi {
            for x in 0..twi {
                if snap[idx(x, y)] != T_OPEN {
                    continue;
                }
                let touches_high = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(dx, dy)| inb(x + dx, y + dy) && snap[idx(x + dx, y + dy)] == T_HIGH);
                if touches_high {
                    t[idx(x, y)] = T_CLIFF;
                }
            }
        }
        // Ramps: punch a passable gap through each plateau's ring.
        for &(cx, cy, hw, hh, side) in &specs {
            for &(mx, my) in &[(cx, cy), (twi - 1 - cx, thi - 1 - cy)] {
                let (rx, ry) = match side & 3 {
                    0 => (mx, my + hh + 1),
                    1 => (mx, my - hh - 1),
                    2 => (mx + hw + 1, my),
                    _ => (mx - hw - 1, my),
                };
                if inb(rx, ry) {
                    t[idx(rx, ry)] = T_RAMP;
                }
            }
        }
        // Keep bases & mineral lines (and their surroundings) on open ground.
        let clears: Vec<(V2, f32)> = self
            .ents
            .iter()
            .filter_map(|e| match e.kind {
                Kind::Base => Some((e.pos, 210.0)),
                Kind::Mineral => Some((e.pos, 95.0)),
                _ => None,
            })
            .collect();
        for (c, r) in clears {
            let rt = (r / TCELL).ceil() as i32;
            let ccx = (c.x / TCELL) as i32;
            let ccy = (c.y / TCELL) as i32;
            for y in (ccy - rt)..=(ccy + rt) {
                for x in (ccx - rt)..=(ccx + rt) {
                    if inb(x, y) {
                        let dx = (x - ccx) as f32 * TCELL;
                        let dy = (y - ccy) as f32 * TCELL;
                        if dx * dx + dy * dy <= r * r {
                            t[idx(x, y)] = T_OPEN;
                        }
                    }
                }
            }
        }
        self.terrain = t;
        // Guarantee connectivity: every faction's base must reach faction 0's
        // base (so the whole map is one connected region). Carve a corridor for
        // any pair the dice walled off.
        let bases: Vec<V2> = (0..self.factions)
            .filter_map(|fi| self.first_base(Team::from_idx(fi)))
            .map(|i| self.ents[i].pos)
            .collect();
        if let Some(&hub) = bases.first() {
            for &b in bases.iter().skip(1) {
                if self.find_path(hub, b).is_none() {
                    self.carve_corridor(hub, b);
                }
            }
        }
        self.has_cliffs = self.terrain.iter().any(|&c| c == T_CLIFF);
    }

    /// Open every cliff tile along the segment a->b (and its immediate sides).
    fn carve_corridor(&mut self, a: V2, b: V2) {
        let d = b.sub(a);
        let steps = (d.len() / (TCELL * 0.4)).ceil() as i32;
        for i in 0..=steps {
            let p = a.add(d.scale(i as f32 / steps.max(1) as f32));
            let cx = (p.x / TCELL) as i32;
            let cy = (p.y / TCELL) as i32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (x, y) = (cx + dx, cy + dy);
                    if x >= 0 && y >= 0 && (x as usize) < self.tw && (y as usize) < self.th {
                        let i = y as usize * self.tw + x as usize;
                        if self.terrain[i] == T_CLIFF {
                            self.terrain[i] = T_OPEN;
                        }
                    }
                }
            }
        }
    }

    /// Wipe terrain back to open ground — used by tests that exercise movement
    /// or combat logic in isolation, independent of the procedural map.
    #[cfg(test)]
    pub fn flatten_terrain(&mut self) {
        for c in self.terrain.iter_mut() {
            *c = T_OPEN;
        }
        self.has_cliffs = false;
    }

    /// Snap a point out of a cliff to the nearest reachable cell centre.
    pub fn snap_open(&self, p: V2) -> V2 {
        if !self.has_cliffs || !self.blocked_pt(p) {
            return p;
        }
        let cx = (p.x / TCELL) as i32;
        let cy = (p.y / TCELL) as i32;
        if let Some((nx, ny)) = self.nearest_passable(cx, cy) {
            return v2((nx as f32 + 0.5) * TCELL, (ny as f32 + 0.5) * TCELL);
        }
        p
    }

    /// Push a point out of any cliff it ended up inside, trying to keep one axis.
    fn unstick(&self, from: V2, to: V2) -> V2 {
        if !self.blocked_pt(to) {
            return to;
        }
        let sx = v2(to.x, from.y);
        if !self.blocked_pt(sx) {
            return sx;
        }
        let sy = v2(from.x, to.y);
        if !self.blocked_pt(sy) {
            return sy;
        }
        from
    }

    // ---- fog of war -------------------------------------------------------

    fn update_fog(&mut self) {
        // Recompute fog for EVERY active faction: the local player views their
        // own grid, and each AI reads its own. Computing all of them keeps AI
        // fog (which feeds the sim) deterministic across peers.
        for fi in 0..self.factions {
            self.fog_pass(Team::from_idx(fi));
        }
    }

    /// Recompute one faction's visibility grid (its slice of `vis`).
    fn fog_pass(&mut self, team: Team) {
        let cells = self.fog_w * self.fog_h;
        let base = team.idx() * cells;
        // Last frame's visible cells fade to "explored" (remembered but dim).
        for v in &mut self.vis[base..base + cells] {
            if *v == 2 {
                *v = 1;
            }
        }
        let cs = self.fog_cell;
        let (fw, fh) = (self.fog_w as i32, self.fog_h as i32);
        let fog_w = self.fog_w;
        let (tw, th) = (self.tw, self.th);
        // Snapshot sight sources, with a vision bonus for high-ground spotters.
        let srcs: Vec<(V2, f32, u8)> = self
            .ents
            .iter()
            .filter(|e| e.team == team)
            .map(|e| {
                let elev = self.elev_at(e.pos);
                let s = sight(e.kind) * if elev == 1 { 1.3 } else { 1.0 };
                (e.pos, s, elev)
            })
            .collect();
        // Read terrain immutably alongside the mutable visibility slice (disjoint
        // fields), so we can enforce the high-ground sight rule.
        let terrain = &self.terrain;
        let grid = &mut self.vis[base..base + cells];
        const CLOSE2: i32 = 2; // cells^2: low ground only sees onto a plateau point-blank
        for (pos, s, se) in srcs {
            if s <= 0.0 {
                continue;
            }
            let cx = (pos.x / cs) as i32;
            let cy = (pos.y / cs) as i32;
            let rc = (s / cs).ceil() as i32;
            let rc2 = (s / cs) * (s / cs);
            for dy in -rc..=rc {
                let gy = cy + dy;
                if gy < 0 || gy >= fh {
                    continue;
                }
                for dx in -rc..=rc {
                    let d2 = dx * dx + dy * dy;
                    if d2 as f32 > rc2 {
                        continue;
                    }
                    let gx = cx + dx;
                    if gx < 0 || gx >= fw {
                        continue;
                    }
                    // You can't see up onto high ground from below except up close.
                    if se == 0 && d2 > CLOSE2 {
                        let txi = (((gx as f32 + 0.5) * cs) / TCELL) as usize;
                        let tyi = (((gy as f32 + 0.5) * cs) / TCELL) as usize;
                        if txi < tw && tyi < th && terrain[tyi * tw + txi] == T_HIGH {
                            continue;
                        }
                    }
                    grid[gy as usize * fog_w + gx as usize] = 2;
                }
            }
        }
    }

    /// Visibility at a world point for the LOCAL player: 0 unseen, 1 explored,
    /// 2 visible. (Renderer and player-facing queries use this.)
    pub fn vis_at(&self, p: V2) -> u8 {
        self.team_vis_at(self.my_team, p)
    }

    /// Visibility at a world point for a specific faction (the AI reads its own).
    pub fn team_vis_at(&self, team: Team, p: V2) -> u8 {
        let idx = team.idx();
        if idx >= self.factions {
            return 0;
        }
        let gx = (p.x / self.fog_cell) as i32;
        let gy = (p.y / self.fog_cell) as i32;
        if gx < 0 || gy < 0 || gx as usize >= self.fog_w || gy as usize >= self.fog_h {
            return 0;
        }
        let cells = self.fog_w * self.fog_h;
        self.vis[idx * cells + gy as usize * self.fog_w + gx as usize]
    }

    pub fn clamp_cam(&mut self, sw: f32, sh: f32) {
        self.cam.x = self.cam.x.clamp(0.0, (self.world_w - sw).max(0.0));
        self.cam.y = self.cam.y.clamp(0.0, (self.world_h - sh).max(0.0));
    }

    // ---- the main tick ----------------------------------------------------

    pub fn update(&mut self, dt: f32) {
        if self.match_over {
            // Match decided: let tracers/particles/messages finish, units idle.
            // (Frozen identically on every peer, so checksums stay matched.)
            self.decay(dt);
            self.update_particles(dt);
            return;
        }
        self.time += dt;
        self.attack_warn -= dt;

        // Run each AI-controlled faction's brain (deterministic on every peer).
        // Rotate the order each tick so no faction — in particular the human's
        // slot 0 — is permanently the first or last to react.
        let rot = (self.time * 60.0 + 0.5) as usize;
        for k in 0..self.factions {
            let fi = (k + rot) % self.factions;
            if self.is_ai[fi] {
                crate::ai::ai_update(self, dt, Team::from_idx(fi));
            }
        }

        self.production(dt);
        let mut dmg: Vec<(usize, f32)> = Vec::new();
        self.decide(dt, &mut dmg);
        self.apply_damage(&mut dmg);
        self.rebuild_block_grid(); // buildings may have spawned this tick
        self.movement(dt);
        self.cleanup();
        self.decay(dt);
        self.update_particles(dt);
        self.update_fog();
        self.check_over();
    }

    fn production(&mut self, dt: f32) {
        let n = self.ents.len();
        for i in 0..n {
            // Construction ramp for buildings being assembled.
            if self.ents[i].build_left > 0.0 {
                let k = self.ents[i].kind;
                let total = build_time(k);
                self.ents[i].build_left -= dt;
                let prog = 1.0 - (self.ents[i].build_left / total).clamp(0.0, 1.0);
                self.ents[i].hp = self.ents[i].max_hp * (0.12 + 0.88 * prog);
                if self.ents[i].build_left <= 0.0 {
                    self.ents[i].build_left = 0.0;
                    self.ents[i].hp = self.ents[i].max_hp;
                    let bp = self.ents[i].pos;
                    self.emit(bp, crate::gfx::rgb(200, 195, 180), 18, 110.0, 3.2, 0.6, -25.0, 2.0, false);
                    self.emit(bp, crate::gfx::rgb(150, 230, 170), 8, 90.0, 2.0, 0.4, -10.0, 4.0, true);
                    self.shock(bp, 40.0, 0.4, crate::gfx::rgb(150, 230, 170));
                    if self.ents[i].team == self.my_team {
                        self.msg("BUILDING ONLINE");
                        self.sfx_at(Sfx::Build, bp);
                    }
                }
                continue;
            }
            if self.ents[i].queue.is_empty() {
                continue;
            }
            let front = self.ents[i].queue[0];
            self.ents[i].train_timer += dt;
            if self.ents[i].train_timer >= build_time(front) {
                self.ents[i].train_timer = 0.0;
                self.ents[i].queue.remove(0);
                let team = self.ents[i].team;
                let bpos = self.ents[i].pos;
                let rally = self.ents[i].rally;
                let rally_set = self.ents[i].rally_set;
                let dir = rally.sub(bpos).norm();
                let spawn_at = bpos.add(dir.scale(radius(self.ents[i].kind) + radius(front) + 4.0));
                let id = self.spawn(front, team, spawn_at);
                if front == Kind::Worker {
                    // Honor an explicit rally: mine a patch under it, else walk
                    // there. With no rally set, fall back to auto-mining.
                    let on_mineral = self
                        .nearest_mineral(rally)
                        .filter(|&m| self.ents[m].pos.dist(rally) < radius(Kind::Mineral) + 26.0);
                    if rally_set {
                        match on_mineral {
                            Some(m) => {
                                let mid = self.ents[m].id;
                                self.ents[id].order = Order::Gather(mid);
                            }
                            None => self.ents[id].order = Order::Move(rally),
                        }
                    } else {
                        self.auto_gather(id);
                    }
                } else {
                    self.ents[id].order = Order::AttackMove(rally);
                }
                if team == self.my_team {
                    self.msg(match front {
                        Kind::Worker => "WORKER READY",
                        Kind::Tank => "TANK READY",
                        _ => "SOLDIER READY",
                    });
                    self.sfx_at(Sfx::Train, bpos);
                }
            }
        }
    }

    fn decide(&mut self, dt: f32, dmg: &mut Vec<(usize, f32)>) {
        let n = self.ents.len();
        for i in 0..n {
            if self.ents[i].cooldown > 0.0 {
                self.ents[i].cooldown -= dt;
            }
            let kind = self.ents[i].kind;
            if !is_mover(kind) {
                continue;
            }
            let team = self.ents[i].team;
            let pos = self.ents[i].pos;
            let order = self.ents[i].order;
            self.ents[i].goal = None;

            // Resolve a combat target: explicit order, or auto-acquire.
            let target = match order {
                Order::Attack(tid) => self.index_of(tid).filter(|&j| self.ents[j].hp > 0.0),
                // Auto-engage only while holding (idle) or attack-moving. A plain
                // Move is obeyed strictly — so you can always reposition or pull
                // a unit out of a fight; it won't re-acquire on its own.
                Order::AttackMove(_) | Order::Idle => {
                    let a = aggro(kind);
                    if a > 0.0 {
                        self.nearest_enemy(team, pos, a)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(j) = target {
                let tpos = self.ents[j].pos;
                let mut reach = atk_range(kind) + radius(kind) + radius(self.ents[j].kind);
                // High-ground edge: firing downhill reaches a little farther.
                if self.has_cliffs && atk_range(kind) > 10.0 && self.elev_at(pos) > self.elev_at(tpos) {
                    reach += 22.0;
                }
                let dist = pos.dist(tpos);
                let mn = min_range(kind);
                if dist < mn {
                    // A mortar can't fire point-blank — fall back out of the dead zone.
                    let away = tpos.sub(pos);
                    let dir = if away.len_sq() > 1.0 { away.norm() } else { v2(0.0, -1.0) };
                    self.ents[i].facing = dir;
                    self.ents[i].goal = Some(pos.sub(dir.scale(mn + 40.0)));
                } else if dist <= reach {
                    // In range: face the target, then fire — or, for a Sapper, blow up.
                    let f = tpos.sub(pos);
                    if f.len_sq() > 1.0 {
                        self.ents[i].facing = f.norm();
                    }
                    if kind == Kind::Sapper {
                        // Suicide blast: hit every nearby enemy, hardest at the centre,
                        // then die (cleanup runs the explosion).
                        let (blast, bd) = sapper_blast();
                        let b2 = blast * blast;
                        for k2 in 0..self.ents.len() {
                            let e = &self.ents[k2];
                            if e.hp <= 0.0 || e.team == team || e.team == Team::Neutral {
                                continue;
                            }
                            let dd = e.pos.dist_sq(pos);
                            if dd <= b2 {
                                dmg.push((k2, bd * (1.0 - 0.55 * (dd / b2))));
                            }
                        }
                        self.ents[i].hp = 0.0;
                    } else if self.ents[i].cooldown <= 0.0 {
                        self.ents[i].cooldown = atk_cd(kind);
                        let dm = damage(kind);
                        if let Some((reach, cone_cos)) = flame(kind) {
                            // Flamethrower: wash the whole cone in front with fire,
                            // damaging every enemy inside the arc.
                            let dir = self.ents[i].facing;
                            let r2 = reach * reach;
                            for k2 in 0..self.ents.len() {
                                let e = &self.ents[k2];
                                if e.hp <= 0.0 || e.team == team || e.team == Team::Neutral {
                                    continue;
                                }
                                let to = e.pos.sub(pos);
                                let d2 = to.len_sq();
                                if d2 > r2 {
                                    continue;
                                }
                                let n = to.norm();
                                if n.x * dir.x + n.y * dir.y >= cone_cos || d2 < 1.0 {
                                    dmg.push((k2, dm));
                                }
                            }
                            let nozzle = pos.add(dir.scale(radius(kind) + 3.0));
                            self.flame_fx(nozzle, dir, reach);
                            if self.time - self.last_shot_snd > 0.07 {
                                self.last_shot_snd = self.time;
                                self.sfx_at(Sfx::Flame, pos);
                            }
                        } else {
                            let sp = splash(kind);
                            if sp > 0.0 {
                                // Splash: damage every enemy near the impact point.
                                let sp2 = sp * sp;
                                for k2 in 0..self.ents.len() {
                                    let e = &self.ents[k2];
                                    if e.hp > 0.0
                                        && e.team != team
                                        && e.team != Team::Neutral
                                        && e.pos.dist_sq(tpos) <= sp2
                                    {
                                        dmg.push((k2, dm));
                                    }
                                }
                            } else {
                                dmg.push((j, dm));
                            }
                            if is_combat(kind) {
                                // Tracer-firing weapons: soldier rifle, tank cannon, raider gun.
                                let col = if team == Team::Player {
                                    crate::gfx::rgb(150, 220, 255)
                                } else {
                                    crate::gfx::rgb(255, 180, 120)
                                };
                                self.tracers.push(Tracer {
                                    a: pos,
                                    b: tpos,
                                    life: if kind == Kind::Tank { 0.12 } else { 0.08 },
                                    color: col,
                                });
                                // Weapon sound — tank cannons always boom; the rattle
                                // of guns is throttled so it doesn't blare.
                                if kind == Kind::Tank {
                                    self.sfx_at(Sfx::TankShot, pos);
                                } else if self.time - self.last_shot_snd > 0.045 {
                                    self.last_shot_snd = self.time;
                                    self.sfx_at(Sfx::Shot, pos);
                                }
                                // Muzzle flash at the barrel.
                                let muzzle = pos.add(self.ents[i].facing.scale(radius(kind) + 2.0));
                                let mcount = if kind == Kind::Tank { 4 } else { 2 };
                                self.emit(muzzle, crate::gfx::rgb(255, 240, 200), mcount, 70.0, 1.6, 0.12, 0.0, 8.0, true);
                                if kind == Kind::Tank {
                                    // Bright impact burst from the splash round.
                                    self.emit(tpos, crate::gfx::rgb(255, 230, 150), 10, 150.0, 2.4, 0.32, 40.0, 5.0, true);
                                    self.shock(tpos, 30.0, 0.22, crate::gfx::rgb(255, 220, 150));
                                    self.add_shake(self.view_gain(tpos));
                                }
                            } else if kind == Kind::Worker {
                                // Improvised melee — a chip of sparks at the point of
                                // contact; workers carry no gun, so no tracer fires.
                                self.emit(tpos, crate::gfx::rgb(255, 220, 160), 3, 60.0, 1.3, 0.18, 26.0, 5.0, true);
                            } else if kind == Kind::Mortar {
                                // A lobbed shell: a launch puff at the tube, then a
                                // wide burst + shockwave where it lands downrange.
                                let dir = self.ents[i].facing;
                                let tube = pos.sub(dir.scale(0.0)).add(dir.scale(radius(kind) + 4.0));
                                self.emit(tube, crate::gfx::rgb(255, 235, 190), 4, 90.0, 1.8, 0.16, -40.0, 6.0, true);
                                self.emit(tpos, crate::gfx::rgb(255, 225, 150), 14, 170.0, 2.6, 0.4, 60.0, 4.0, true);
                                self.emit(tpos, crate::gfx::rgb(255, 150, 70), 10, 120.0, 2.2, 0.5, 40.0, 3.0, true);
                                self.shock(tpos, splash(kind), 0.3, crate::gfx::rgb(255, 210, 140));
                                self.flash(0.12 * self.view_gain(tpos), crate::gfx::rgb(255, 180, 90));
                                self.add_shake(1.4 * self.view_gain(tpos));
                                self.sfx_at(Sfx::TankShot, pos);
                            }
                        }
                    }
                    self.ents[i].goal = None;
                } else {
                    self.ents[i].goal = Some(tpos);
                }
                continue;
            }

            // No combat target: carry out the standing order.
            match order {
                Order::Attack(_) => {
                    // Target gone — on to the next queued order, or idle.
                    self.advance_order(i);
                }
                Order::Move(p) | Order::AttackMove(p) => {
                    if pos.dist(p) < 6.0 {
                        self.advance_order(i);
                    } else {
                        self.ents[i].goal = Some(p);
                    }
                }
                Order::Gather(mid) => self.run_gather(i, mid, dt),
                Order::Repair(bid) => self.run_repair(i, bid, dt),
                Order::Build(bk, site) => {
                    if pos.dist(site) <= radius(bk) + radius(kind) + 6.0 {
                        // Raise the building in construction state.
                        let bi = self.spawn(bk, team, site);
                        self.ents[bi].build_left = build_time(bk);
                        self.ents[bi].hp = self.ents[bi].max_hp * 0.12;
                        self.ents[bi].rally = site.add(v2(0.0, radius(bk) + 30.0));
                        // On to the next chained build, or back to mining.
                        if self.ents[i].build_queue.is_empty() {
                            self.auto_gather(i);
                        } else {
                            let (nk, ns) = self.ents[i].build_queue.remove(0);
                            self.ents[i].order = Order::Build(nk, ns);
                        }
                    } else {
                        self.ents[i].goal = Some(site);
                    }
                }
                Order::Idle => {}
            }
        }
    }

    /// Finished the current order: pull the next queued waypoint, or go idle.
    fn advance_order(&mut self, i: usize) {
        self.ents[i].goal = None;
        if self.ents[i].order_queue.is_empty() {
            self.ents[i].order = Order::Idle;
        } else {
            self.ents[i].order = self.ents[i].order_queue.remove(0);
        }
    }

    /// A worker repairs a friendly building: walk to it, then restore HP over
    /// time, spending a fraction of the build cost as it goes. Pauses if the bank
    /// runs dry; returns to mining (or the next order) once the building is whole.
    fn run_repair(&mut self, i: usize, bid: u32, dt: f32) {
        let pos = self.ents[i].pos;
        let team = self.ents[i].team;
        let bi = self.index_of(bid).filter(|&b| {
            self.ents[b].team == team && is_building(self.ents[b].kind) && self.ents[b].hp > 0.0
        });
        let Some(bi) = bi else {
            self.advance_order(i);
            return;
        };
        let mhp = self.ents[bi].max_hp;
        if self.ents[bi].hp >= mhp {
            // Whole again — next queued order, else back to the mineral line.
            if self.ents[i].order_queue.is_empty() {
                self.auto_gather(i);
            } else {
                self.advance_order(i);
            }
            return;
        }
        let bpos = self.ents[bi].pos;
        let reach = radius(Kind::Worker) + radius(self.ents[bi].kind) + 6.0;
        if pos.dist(bpos) > reach {
            self.ents[i].goal = Some(bpos);
            return;
        }
        self.ents[i].goal = None;
        let kind = self.ents[bi].kind;
        let heal = mhp / REPAIR_SECONDS * dt;
        self.ents[i].repair_owed += (heal / mhp) * cost(kind) as f32 * REPAIR_COST_FRAC;
        // Spend whole minerals as the debt accrues; each one throws a repair spark.
        while self.ents[i].repair_owed >= 1.0 {
            if self.team_min(team) == 0 {
                return; // broke: hold, don't heal for free
            }
            self.spend(team, 1);
            self.ents[i].repair_owed -= 1.0;
            self.emit(bpos, crate::gfx::rgb(150, 235, 205), 2, 70.0, 1.5, 0.3, 26.0, 4.0, true);
        }
        self.ents[bi].hp = (self.ents[bi].hp + heal).min(mhp);
    }

    fn run_gather(&mut self, i: usize, mid: u32, dt: f32) {
        let pos = self.ents[i].pos;
        let team = self.ents[i].team;
        if self.ents[i].carry < CARRY {
            // Head to the patch and mine. Only re-pick a patch once the assigned
            // one is exhausted; a stable assignment keeps the field saturated
            // without thrashing.
            let wid = self.ents[i].id;
            let mi = self
                .index_of(mid)
                .filter(|&m| self.ents[m].kind == Kind::Mineral && self.ents[m].minerals > 0)
                .or_else(|| self.pick_patch(pos, wid));
            let Some(mi) = mi else {
                self.ents[i].order = Order::Idle;
                return;
            };
            // Keep the order pointed at the patch we actually found.
            self.ents[i].order = Order::Gather(self.ents[mi].id);
            let mpos = self.ents[mi].pos;
            if pos.dist(mpos) <= radius(Kind::Worker) + radius(Kind::Mineral) + 5.0 {
                self.ents[i].mine_timer += dt;
                if self.ents[i].mine_timer >= MINE_TIME {
                    self.ents[i].mine_timer = 0.0;
                    let got = CARRY.min(self.ents[mi].minerals);
                    self.ents[mi].minerals -= got;
                    self.ents[i].carry = got;
                    self.emit(mpos, crate::gfx::rgb(160, 245, 255), 5, 80.0, 1.7, 0.5, 30.0, 4.0, true);
                }
                self.ents[i].goal = None;
            } else {
                // Aim for the rim of the patch on our side, not the dead centre,
                // so two workers sharing a patch settle on different edges
                // instead of grinding into the same point.
                let dir = pos.sub(mpos);
                let aim = if dir.len() > 1.0 {
                    mpos.add(dir.norm().scale(radius(Kind::Mineral) + radius(Kind::Worker) + 2.0))
                } else {
                    mpos
                };
                self.ents[i].goal = Some(aim);
            }
        } else {
            // Haul it home.
            let Some(bi) = self.nearest_base(team, pos) else {
                self.ents[i].goal = None;
                return;
            };
            let bpos = self.ents[bi].pos;
            if pos.dist(bpos) <= radius(Kind::Worker) + radius(Kind::Base) + 6.0 {
                self.gain(team, self.ents[i].carry);
                self.ents[i].carry = 0;
                self.ents[i].goal = None;
            } else {
                self.ents[i].goal = Some(bpos);
            }
        }
    }

    fn apply_damage(&mut self, dmg: &mut Vec<(usize, f32)>) {
        for &(i, d) in dmg.iter() {
            if i >= self.ents.len() {
                continue;
            }
            self.ents[i].hp -= d;
            self.ents[i].flash = 0.1;
            let pos = self.ents[i].pos;
            // Impact sparks at the point of contact.
            self.emit(pos, crate::gfx::rgb(255, 235, 170), 3, 90.0, 1.6, 0.2, 50.0, 6.0, true);
            if self.ents[i].team == self.my_team && self.attack_warn <= 0.0 {
                self.attack_warn = 6.0;
                self.msg("UNDER ATTACK!");
                self.sfx(Sfx::Alarm);
            }
        }
    }

    fn movement(&mut self, dt: f32) {
        let n = self.ents.len();
        // Step toward goals, routing around terrain via waypoints.
        for i in 0..n {
            let kind = self.ents[i].kind;
            if !is_mover(kind) {
                continue;
            }
            let pos = self.ents[i].pos;
            let Some(g) = self.ents[i].goal else {
                continue;
            };

            // Decide what to steer at this tick: the next path waypoint, or the
            // goal directly when the straight line is clear. We path around both
            // cliffs and buildings, so this runs whenever there are obstacles
            // (there are always buildings) — A* only fires if the line is blocked.
            self.ents[i].repath -= dt;
            if self.ents[i].repath <= 0.0 {
                self.ents[i].repath = 0.4;
                // Within a short hop, approach the goal directly THROUGH soft
                // obstacles (the patch it mines, its own base, a build site) so
                // pathing doesn't circle the target forever — but a hard ridge
                // must always be routed around, even a short hop away, or units
                // grind helplessly into the cliff face.
                let near = pos.dist(g) <= TCELL * 1.5;
                let need_path = if near {
                    self.line_blocked_cliff(pos, g)
                } else {
                    self.line_blocked(pos, g)
                };
                self.ents[i].path = if need_path {
                    self.find_path(pos, g).unwrap_or_default()
                } else {
                    Vec::new()
                };
            }
            // Pop waypoints we've reached.
            while let Some(&wp) = self.ents[i].path.first() {
                if pos.dist(wp) < TCELL * 0.5 {
                    self.ents[i].path.remove(0);
                } else {
                    break;
                }
            }
            let steer = *self.ents[i].path.first().unwrap_or(&g);

            let to = steer.sub(pos);
            let d = to.len();
            if d > 1.0 {
                self.ents[i].facing = to.norm(); // face the way we're heading
            }
            let step = speed(kind) * dt;
            let cand = if d > step { pos.add(to.scale(step / d)) } else { steer };
            self.ents[i].pos = if self.has_cliffs { self.unstick(pos, cand) } else { cand };
        }
        // Local collision avoidance: push overlapping bodies apart.
        for i in 0..n {
            if !is_mover(self.ents[i].kind) {
                continue;
            }
            let ri = radius(self.ents[i].kind);
            let mut push = v2(0.0, 0.0);
            let pi = self.ents[i].pos;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let rj = radius(self.ents[j].kind);
                let min_d = ri + rj;
                let diff = pi.sub(self.ents[j].pos);
                let d = diff.len();
                if d > 0.001 && d < min_d {
                    let overlap = min_d - d;
                    let w = if is_mover(self.ents[j].kind) { 0.5 } else { 1.0 };
                    push = push.add(diff.scale((overlap * w) / d));
                }
            }
            let pushed = pi.add(push);
            // Don't let separation shove a body into a cliff.
            self.ents[i].pos = if self.has_cliffs { self.unstick(pi, pushed) } else { pushed };
            // Stay on the map.
            let r = radius(self.ents[i].kind);
            self.ents[i].pos.x = self.ents[i].pos.x.clamp(r, self.world_w - r);
            self.ents[i].pos.y = self.ents[i].pos.y.clamp(r, self.world_h - r);
        }
    }

    fn cleanup(&mut self) {
        // Gather everything dying this frame for death effects + the scoreboard.
        let dying: Vec<(V2, Kind, Team)> = self
            .ents
            .iter()
            .filter(|e| if e.kind == Kind::Mineral { e.minerals == 0 } else { e.hp <= 0.0 })
            .map(|e| (e.pos, e.kind, e.team))
            .collect();
        self.kills += dying
            .iter()
            .filter(|(_, k, t)| *t == Team::Enemy && is_mover(*k))
            .count() as u32;
        for (pos, kind, team) in dying {
            self.death_fx(pos, kind, team);
        }
        self.ents.retain(|e| {
            if e.kind == Kind::Mineral {
                e.minerals > 0
            } else {
                e.hp > 0.0
            }
        });
    }

    fn decay(&mut self, dt: f32) {
        for t in self.tracers.iter_mut() {
            t.life -= dt;
        }
        self.tracers.retain(|t| t.life > 0.0);
        for p in self.pings.iter_mut() {
            p.1 -= dt;
        }
        self.pings.retain(|p| p.1 > 0.0);
        for m in self.messages.iter_mut() {
            m.1 -= dt;
        }
        self.messages.retain(|m| m.1 > 0.0);
        for e in self.ents.iter_mut() {
            if e.flash > 0.0 {
                e.flash -= dt;
            }
        }
    }

    /// True while faction `t` still holds at least one building (of any kind).
    /// Losing your Command Center isn't game over if a Barracks/Factory/Depot
    /// stands — you fight on and can rebuild. But the moment your last building
    /// falls you're out, so a lone surviving unit isn't a tedious end-game chase.
    pub fn faction_alive(&self, t: Team) -> bool {
        self.ents.iter().any(|e| e.team == t && is_building(e.kind))
    }

    fn check_over(&mut self) {
        // Count surviving factions (a faction is out once it has nothing left).
        let mut alive = 0usize;
        let mut last = Team::Neutral;
        for fi in 0..self.factions {
            let t = Team::from_idx(fi);
            if self.faction_alive(t) {
                alive += 1;
                last = t;
            }
        }
        let my_alive = self.faction_alive(self.my_team);
        // Local result for the UI: defeat the moment my faction is wiped, victory
        // when I'm the sole survivor.
        if !my_alive {
            self.over = -1;
        } else if alive <= 1 {
            self.over = 1;
        }
        // The match (and the sim) ends only when one faction remains — except in
        // single-player, where losing my faction ends it immediately.
        if alive <= 1 || (!my_alive && !self.versus) {
            self.match_over = true;
        }
        let _ = last;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_player_worker(w: &World) -> usize {
        w.ents
            .iter()
            .position(|e| e.team == Team::Player && e.kind == Kind::Worker)
            .expect("player should start with workers")
    }

    #[test]
    fn defeat_when_the_last_building_falls() {
        let mut w = World::new(2);
        // A second building means losing the Command Center isn't the end.
        w.spawn(Kind::Barracks, Team::Player, v2(700.0, 1000.0));
        for e in w.ents.iter_mut() {
            if e.team == Team::Player && e.kind == Kind::Base {
                e.hp = -1.0;
            }
        }
        w.update(1.0 / 60.0);
        assert!(w.faction_alive(Team::Player), "a standing Barracks keeps you in");
        assert_eq!(w.over, 0, "losing only the CC must not be a defeat");
        // Raze the last building. Workers still live, but with no buildings the
        // game ends — no tedious hunt for a lone surviving unit.
        for e in w.ents.iter_mut() {
            if e.team == Team::Player && e.kind == Kind::Barracks {
                e.hp = -1.0;
            }
        }
        w.update(1.0 / 60.0);
        assert!(
            w.ents.iter().any(|e| e.team == Team::Player && e.kind == Kind::Worker),
            "workers are still alive on the map"
        );
        assert!(!w.faction_alive(Team::Player));
        assert_eq!(w.over, -1, "no buildings left = defeat, even with units alive");
    }

    #[test]
    fn train_presses_balance_across_selected_buildings() {
        let mut w = World::new(2);
        let mut ids = Vec::new();
        for k in 0..3 {
            let b = w.spawn(Kind::Barracks, Team::Player, v2(600.0 + k as f32 * 120.0, 1000.0));
            w.ents[b].selected = true;
            ids.push(w.ents[b].id);
        }
        // Each press lands on the least-loaded selected building.
        for _ in 0..6 {
            let id = w.least_loaded_selected(Kind::Barracks).expect("a ready barracks");
            let bi = w.index_of(id).unwrap();
            w.ents[bi].queue.push(Kind::Soldier);
        }
        for &id in &ids {
            assert_eq!(
                w.ents[w.index_of(id).unwrap()].queue.len(),
                2,
                "the six units should spread two-per-barracks"
            );
        }
        // An unselected building of the same kind never gets production.
        let other = w.spawn(Kind::Barracks, Team::Player, v2(2200.0, 300.0));
        let _ = other;
        let id = w.least_loaded_selected(Kind::Barracks).unwrap();
        assert!(ids.contains(&id), "only selected buildings receive trained units");
    }

    #[test]
    fn ctrl_click_chains_multiple_builds() {
        let mut w = World::new(2);
        w.flatten_terrain();
        w.minerals[0] = 99_999;
        let wi = a_player_worker(&w);
        w.ents[wi].pos = v2(1000.0, 500.0);
        let sites = [v2(1000.0, 500.0), v2(1000.0, 680.0), v2(1000.0, 860.0)];
        // First a plain build, then two chained ones.
        assert!(w.order_build(wi, Kind::Depot, sites[0]), "site 0 is buildable");
        assert!(w.queue_build(wi, Kind::Depot, sites[1]), "site 1 is buildable");
        assert!(w.queue_build(wi, Kind::Depot, sites[2]), "site 2 is buildable");
        assert_eq!(w.ents[wi].build_queue.len(), 2, "two builds queued behind the first");
        // A chained site too close to a pending one is rejected.
        assert!(!w.queue_build(wi, Kind::Depot, sites[0].add(v2(10.0, 0.0))), "overlap rejected");
        // Walk the worker onto each site so it raises them in turn.
        for &s in &sites {
            w.ents[wi].pos = s;
            w.update(1.0 / 60.0);
        }
        let depots = w
            .ents
            .iter()
            .filter(|e| e.team == Team::Player && e.kind == Kind::Depot)
            .count();
        assert_eq!(depots, 3, "all three chained buildings should rise");
        assert!(
            !matches!(w.ents[wi].order, Order::Build(_, _)),
            "the worker is freed once the chain is done"
        );
    }

    #[test]
    fn build_chain_is_robust_to_clicking_away() {
        let mut w = World::new(2);
        w.flatten_terrain();
        w.minerals[0] = 9999;
        let wi = a_player_worker(&w);
        w.ents[wi].pos = v2(1000.0, 500.0);
        let wid = w.ents[wi].id;
        let s0 = v2(1000.0, 500.0);
        let s1 = v2(1000.0, 680.0);
        let m0 = w.minerals[0];
        // Place two via the command path. The second is a non-chain "final" click
        // (chain:false) — it must APPEND, not wipe the first.
        w.apply_cmd(Team::Player, &Cmd::Build { worker: wid, kind: Kind::Depot, x: s0.x, y: s0.y, chain: true });
        w.apply_cmd(Team::Player, &Cmd::Build { worker: wid, kind: Kind::Depot, x: s1.x, y: s1.y, chain: false });
        assert!(matches!(w.ents[wi].order, Order::Build(_, _)), "worker started building");
        assert_eq!(w.ents[wi].build_queue.len(), 1, "a plain placement appends, never clears");
        // Deliberately redirecting the worker cancels the whole unbuilt chain and
        // refunds it — no orphaned queue, no lost minerals.
        w.apply_order(Team::Player, &[wid], v2(1500.0, 500.0), false, false);
        assert!(matches!(w.ents[wi].order, Order::Move(_)), "worker takes the move order");
        assert!(w.ents[wi].build_queue.is_empty(), "build chain dropped");
        assert_eq!(w.minerals[0], m0, "both unbuilt depots refunded");
    }

    #[test]
    fn double_click_building_selects_the_nearby_cluster() {
        let mut w = World::new(2);
        w.clear_selection();
        let b0 = w.spawn(Kind::Barracks, Team::Player, v2(600.0, 1000.0));
        let b1 = w.spawn(Kind::Barracks, Team::Player, v2(700.0, 1050.0));
        let b2 = w.spawn(Kind::Barracks, Team::Player, v2(620.0, 1120.0));
        let far = w.spawn(Kind::Barracks, Team::Player, v2(2200.0, 300.0));
        let at = w.ents[b0].pos;
        w.select_type_in_view(at, v2(0.0, 0.0), v2(w.world_w, w.world_h), false);
        assert!(
            w.ents[b0].selected && w.ents[b1].selected && w.ents[b2].selected,
            "double-clicking a building grabs the nearby same-kind cluster"
        );
        assert!(!w.ents[far].selected, "a distant building is not part of the cluster");
    }

    #[test]
    fn tie_key_does_not_favor_the_human() {
        // Entity ids run in spawn order, so the human (slot 0) holds the lowest
        // block. A monotonic tie-break would point every distance tie at it. The
        // hash must scramble that: some higher id beats id 1.
        let k1 = tie_key(1);
        assert!(
            (2..32).any(|id| tie_key(id) < k1),
            "target-acquisition ties still resolve to the lowest id (the human)"
        );
    }

    #[test]
    fn human_start_corner_is_not_fixed() {
        // The human's slot must not be pinned to one (weaker) corner in 3-4p; the
        // seed shuffles the assignment. Across seeds it should land in several.
        let mut quadrants = std::collections::HashSet::new();
        for s in 0..24u64 {
            let w = World::new_match(s.wrapping_mul(2654435761).wrapping_add(1), 4, [false, true, true, true], Team::Player, false);
            let bp = w.ents[w.first_base(Team::Player).unwrap()].pos;
            quadrants.insert(((bp.x > w.world_w * 0.5) as u8, (bp.y > w.world_h * 0.5) as u8));
        }
        assert!(
            quadrants.len() >= 3,
            "human start corner barely varies ({} of 4 quadrants seen)",
            quadrants.len()
        );
    }

    #[test]
    fn forgiving_click_grabs_a_nearby_worker() {
        let mut w = World::new(1);
        let wi = a_player_worker(&w);
        let p = w.ents[wi].pos;
        // Click 6px off the body (smaller than the 7+12 tolerance): still hits.
        w.select_single(p.add(v2(6.0, 0.0)), false);
        assert!(w.ents.iter().any(|e| e.selected && e.kind == Kind::Worker));
    }

    #[test]
    fn empty_click_clears_selection() {
        let mut w = World::new(1);
        w.select_box(v2(0.0, 0.0), v2(WORLD_W, WORLD_H), false);
        assert!(!w.selected_ids().is_empty());
        // Click far from anything.
        w.select_single(v2(WORLD_W - 5.0, 5.0), false);
        assert!(w.selected_ids().is_empty());
    }

    #[test]
    fn control_group_roundtrip() {
        let mut w = World::new(1);
        w.select_box(v2(0.0, 0.0), v2(WORLD_W, WORLD_H), false);
        let ids = w.selected_ids();
        assert!(ids.len() >= 4, "starting workers should be selectable");
        w.clear_selection();
        assert!(w.selected_ids().is_empty());
        w.select_ids(&ids, false);
        assert_eq!(w.selected_ids().len(), ids.len());
        assert!(w.centroid_of_ids(&ids).is_some());
    }

    #[test]
    fn double_click_selects_all_of_type_in_view() {
        let mut w = World::new(1);
        let wi = a_player_worker(&w);
        let p = w.ents[wi].pos;
        w.select_type_in_view(p, v2(0.0, 0.0), v2(WORLD_W, WORLD_H), false);
        let workers = w
            .ents
            .iter()
            .filter(|e| e.team == Team::Player && e.kind == Kind::Worker)
            .count();
        let selected = w.ents.iter().filter(|e| e.selected).count();
        assert_eq!(selected, workers);
        // The base is not a worker, so it must not be caught.
        assert!(w
            .ents
            .iter()
            .all(|e| !(e.selected && e.kind == Kind::Base)));
    }

    #[test]
    fn attack_move_command_moves_unit() {
        let mut w = World::new(7);
        // Isolated player soldier, far from any enemy.
        let s = w.spawn(Kind::Soldier, Team::Player, v2(400.0, 1100.0));
        w.ents[s].selected = true;
        let start_x = w.ents[s].pos.x;
        w.command(v2(900.0, 1100.0), true); // attack-move east
        for _ in 0..120 {
            w.update(1.0 / 60.0);
        }
        let sx = w.ents.iter().find(|e| e.kind == Kind::Soldier).unwrap().pos.x;
        assert!(sx > start_x + 50.0, "attack-move did not move the soldier (x {sx})");
    }

    #[test]
    fn soldier_honors_spawn_rally() {
        let mut w = World::new(7);
        let b = w.spawn(Kind::Barracks, Team::Player, v2(400.0, 1100.0));
        w.ents[b].rally = v2(900.0, 1100.0); // far rally to the east
        w.minerals[0] = 999;
        assert!(w.try_train(b, Kind::Soldier));
        for _ in 0..(60 * 9) {
            w.update(1.0 / 60.0);
        }
        let s = w
            .ents
            .iter()
            .find(|e| e.team == Team::Player && e.kind == Kind::Soldier)
            .expect("a soldier should have been produced");
        assert!(s.pos.x > 600.0, "soldier ignored its rally (x {})", s.pos.x);
    }

    #[test]
    fn ffa_four_factions_all_spawn() {
        let w = World::new_match(99, 4, [true, true, true, true], Team::Player, false);
        for fi in 0..4 {
            assert!(w.faction_alive(Team::from_idx(fi)), "faction {fi} should have a base");
        }
        assert_eq!(w.factions, 4);
        // Bigger map for more players.
        assert!(w.world_w > WORLD_W);
    }

    #[test]
    fn ffa_is_deterministic() {
        // Two identical-seed 4-way games must stay byte-identical (lockstep).
        let cfg = |s| World::new_match(s, 4, [true, true, true, true], Team::Player, false);
        let (mut a, mut b) = (cfg(5), cfg(5));
        for _ in 0..600 {
            a.update(1.0 / 60.0);
            b.update(1.0 / 60.0);
        }
        assert_eq!(a.checksum(), b.checksum(), "same seed must produce identical sims");
    }

    #[test]
    fn ffa_resolves_to_one_winner() {
        // Four AIs fight to the finish; the match ends only when one remains.
        // (versus=true so the single-player "your faction died" shortcut, which
        // would fire when AI faction 0 falls, doesn't end it early.)
        let mut w = World::new_match(3, 4, [true, true, true, true], Team::Player, true);
        let mut ended = false;
        for _ in 0..(60 * 360) {
            w.update(1.0 / 60.0);
            if w.match_over {
                ended = true;
                break;
            }
        }
        // It either crowned a winner or is still going — but never panicked, and
        // if it ended, at most one faction is alive.
        if ended {
            let alive = (0..4).filter(|&fi| w.faction_alive(Team::from_idx(fi))).count();
            assert!(alive <= 1, "match ended with {alive} factions alive");
        }
    }

    #[test]
    fn path_routes_around_buildings() {
        // A wall of buildings blocks the straight line; the unit must route
        // around it rather than jam against it.
        let mut w = World::new(7);
        w.flatten_terrain();
        let u = w.spawn(Kind::Worker, Team::Player, v2(400.0, 600.0));
        let uid = w.ents[u].id;
        let goal = v2(1200.0, 600.0);
        for k in 0..7 {
            w.spawn(Kind::Depot, Team::Player, v2(800.0, 400.0 + k as f32 * 60.0));
        }
        w.ents[u].order = Order::Move(goal);
        for _ in 0..(60 * 20) {
            w.update(1.0 / 60.0);
        }
        let p = w.ents[w.index_of(uid).unwrap()].pos;
        assert!(p.x > 1050.0, "unit should path past the building wall (x {})", p.x);
    }

    #[test]
    fn units_route_around_a_ridge_near_the_goal() {
        // A goal just past a cliff must still be reached by routing around it —
        // the near-goal steer-direct shortcut must not skip a hard ridge.
        let mut w = World::new(7);
        w.flatten_terrain();
        let tw = w.tw;
        for ty in 6..=12 {
            w.terrain[ty * tw + 13] = T_CLIFF;
        }
        w.has_cliffs = true;
        let s = w.spawn(Kind::Soldier, Team::Player, v2(770.0, 540.0));
        let sid = w.ents[s].id;
        let goal = v2(850.0, 540.0); // ~80px away, but the wall sits between
        w.ents[s].order = Order::Move(goal);
        for _ in 0..(60 * 25) {
            w.update(1.0 / 60.0);
        }
        let p = w.ents[w.index_of(sid).unwrap()].pos;
        assert!(p.dist(goal) < 36.0, "unit should round the ridge to the goal (at {p:?})");
    }

    #[test]
    fn worker_repairs_a_damaged_building() {
        let mut w = World::new(2);
        w.flatten_terrain();
        w.minerals[0] = 9999;
        let bp = w.ents[w.first_base(Team::Player).unwrap()].pos;
        let b = w.spawn(Kind::Barracks, Team::Player, bp.add(v2(160.0, 0.0)));
        w.ents[b].hp = w.ents[b].max_hp * 0.4;
        let hurt = w.ents[b].hp;
        let bid = w.ents[b].id;
        let wi = a_player_worker(&w);
        w.ents[wi].pos = w.ents[b].pos.add(v2(50.0, 0.0));
        let wid = w.ents[wi].id;
        // Drop the other workers so mining income doesn't mask the repair cost.
        for e in w.ents.iter_mut() {
            if e.team == Team::Player && e.kind == Kind::Worker && e.id != wid {
                e.hp = -1.0;
            }
        }
        w.apply_order(Team::Player, &[wid], w.ents[b].pos, false, false);
        assert!(matches!(w.ents[wi].order, Order::Repair(_)), "worker takes a repair order");
        let min0 = w.minerals[0];
        for _ in 0..(60 * 6) {
            w.update(1.0 / 60.0);
        }
        let bi = w.index_of(bid).unwrap();
        assert!(w.ents[bi].hp > hurt + 60.0, "building should be repaired (hp {})", w.ents[bi].hp);
        assert!(w.minerals[0] < min0, "repair should cost minerals");
    }

    #[test]
    fn worker_can_melee_an_enemy() {
        let mut w = World::new(2);
        w.flatten_terrain();
        let wi = a_player_worker(&w);
        let wp = w.ents[wi].pos;
        let e = w.spawn(Kind::Soldier, Team::Enemy, wp.add(v2(22.0, 0.0)));
        let eid = w.ents[e].id;
        let hp0 = w.ents[e].hp;
        let wid = w.ents[wi].id;
        w.apply_order(Team::Player, &[wid], w.ents[e].pos, false, false);
        assert!(matches!(w.ents[wi].order, Order::Attack(_)), "worker takes an attack order");
        for _ in 0..(60 * 3) {
            w.update(1.0 / 60.0);
        }
        let damaged = w.index_of(eid).map_or(true, |j| w.ents[j].hp < hp0);
        assert!(damaged, "a worker's melee should hurt the enemy");
    }

    #[test]
    fn shift_queues_unit_waypoints() {
        let mut w = World::new(2);
        w.flatten_terrain();
        let s = w.spawn(Kind::Soldier, Team::Player, v2(900.0, 500.0));
        let sid = w.ents[s].id;
        w.apply_order(Team::Player, &[sid], v2(1200.0, 500.0), false, false);
        w.apply_order(Team::Player, &[sid], v2(1200.0, 900.0), false, true);
        assert_eq!(w.ents[s].order_queue.len(), 1, "the second order should queue");
        for _ in 0..(60 * 30) {
            w.update(1.0 / 60.0);
            if w.ents[w.index_of(sid).unwrap()].pos.dist(v2(1200.0, 900.0)) < 20.0 {
                break;
            }
        }
        let p = w.ents[w.index_of(sid).unwrap()].pos;
        assert!(p.dist(v2(1200.0, 900.0)) < 32.0, "unit should reach the queued waypoint (at {p:?})");
    }

    #[test]
    fn ai_rebuilds_after_losing_its_command_center() {
        let mut w = World::new_match(5, 2, [false, true, false, false], Team::Player, true);
        w.flatten_terrain();
        let ei = w.ents.iter().position(|e| e.team == Team::Enemy && e.kind == Kind::Base).unwrap();
        let ep = w.ents[ei].pos;
        w.spawn(Kind::Barracks, Team::Enemy, ep.add(v2(150.0, 0.0))); // a surviving building
        w.minerals[Team::Enemy.idx()] = 9999;
        for e in w.ents.iter_mut() {
            if e.team == Team::Enemy && e.kind == Kind::Base {
                e.hp = -1.0;
            }
        }
        let mut rebuilt = false;
        for _ in 0..(60 * 45) {
            w.update(1.0 / 60.0);
            if w.ents.iter().any(|e| e.team == Team::Enemy && e.kind == Kind::Base) {
                rebuilt = true;
                break;
            }
        }
        assert!(rebuilt, "AI should rebuild a Command Center after losing it");
    }

    #[test]
    fn units_step_through_a_gap_in_a_building_wall() {
        // The flip side of routing around a wall: a real doorway must stay open.
        // A vertical wall at x=800 with a one-slot gap at y≈570; a unit crossing
        // should thread the gap, not detour around the whole structure. Guards
        // against the block grid over-inflating footprints and fusing the gap.
        let mut w = World::new(7);
        w.flatten_terrain();
        for &y in &[360.0, 420.0, 480.0, 600.0, 660.0, 720.0] {
            w.spawn(Kind::Depot, Team::Player, v2(800.0, y));
        }
        w.update(1.0 / 60.0); // build the block grid
        let from = v2(560.0, 570.0);
        let to = v2(1040.0, 570.0);
        let path = w.find_path(from, to).expect("a path to the far side");
        let mut len = from.dist(path[0]);
        for k in 1..path.len() {
            len += path[k - 1].dist(path[k]);
        }
        // Straight through the doorway is ~480; going around the wall is far
        // longer. A tight bound proves the unit takes the gap.
        assert!(len < 640.0, "unit should step through the gap (path len {len:.0})");
    }

    #[test]
    fn units_route_around_mineral_fields() {
        // A unit crossing a mineral line should path around it, not wedge into a
        // deposit (steering alone stalls against the patch; A* must see it).
        let mut w = World::new(7);
        w.flatten_terrain();
        for k in 0..6 {
            let m = w.spawn(Kind::Mineral, Team::Neutral, v2(800.0, 420.0 + k as f32 * 50.0));
            w.ents[m].minerals = 1000;
        }
        let u = w.spawn(Kind::Soldier, Team::Player, v2(500.0, 600.0));
        let uid = w.ents[u].id;
        w.ents[u].order = Order::Move(v2(1150.0, 600.0));
        for _ in 0..(60 * 20) {
            w.update(1.0 / 60.0);
        }
        let p = w.ents[w.index_of(uid).unwrap()].pos;
        assert!(p.x > 1050.0, "unit should route past the mineral field (x {})", p.x);
    }

    #[test]
    fn base_rally_sends_worker_to_point() {
        let mut w = World::new(7);
        w.flatten_terrain(); // rally logic, not terrain navigation
        let base = w.first_base(Team::Player).unwrap();
        // A rally on open ground a short walk from the base (no mineral nearby).
        let pt = w.ents[base].pos.add(v2(260.0, -90.0));
        w.ents[base].rally = pt;
        w.ents[base].rally_set = true;
        w.minerals[0] = 999;
        assert!(w.try_train(base, Kind::Worker));
        for _ in 0..(60 * 12) {
            w.update(1.0 / 60.0);
        }
        let near = w.ents.iter().any(|e| {
            e.team == Team::Player && e.kind == Kind::Worker && e.pos.dist(pt) < 60.0
        });
        assert!(near, "a rallied worker should walk to the rally point, not mine");
    }

    #[test]
    fn depot_raises_supply_cap() {
        let mut w = World::new(7);
        assert_eq!(w.supply_cap(Team::Player), 11); // command center only
        let d = w.spawn(Kind::Depot, Team::Player, v2(600.0, 1200.0));
        w.ents[d].build_left = 0.0;
        assert_eq!(w.supply_cap(Team::Player), 19);
    }

    #[test]
    fn supply_blocks_overtraining() {
        let mut w = World::new(7);
        w.minerals[0] = 100_000; // only supply can stop us
        let base = w.first_base(Team::Player).unwrap();
        let mut trained = 0;
        for _ in 0..50 {
            if w.try_train(base, Kind::Worker) {
                trained += 1;
            }
        }
        // 4 starting workers already use 4 of 11 supply.
        assert!(trained <= 7, "overtrained past the cap ({trained})");
        assert!(trained >= 1);
    }

    #[test]
    fn tank_splash_hits_multiple() {
        let mut w = World::new(7);
        w.spawn(Kind::Tank, Team::Player, v2(400.0, 1100.0));
        for k in 0..3 {
            w.spawn(Kind::Soldier, Team::Enemy, v2(480.0, 1090.0 + k as f32 * 10.0));
        }
        for _ in 0..30 {
            w.update(1.0 / 60.0);
        }
        let damaged = w
            .ents
            .iter()
            .filter(|e| e.team == Team::Enemy && e.kind == Kind::Soldier && e.hp < e.max_hp)
            .count();
        assert!(damaged >= 2, "tank splash should hit several units ({damaged})");
    }

    #[test]
    fn pyro_cone_burns_a_cluster() {
        // A Pyro should torch several clustered enemies in front of it at once,
        // and its flame is short-ranged.
        let mut w = World::new(7);
        let p = w.spawn(Kind::Pyro, Team::Player, v2(400.0, 1100.0));
        w.ents[p].facing = v2(1.0, 0.0); // pointing at the cluster to its right
        for k in 0..4 {
            w.spawn(Kind::Soldier, Team::Enemy, v2(452.0, 1085.0 + k as f32 * 10.0));
        }
        // A bystander well out of flame range should stay untouched.
        let far = w.spawn(Kind::Soldier, Team::Enemy, v2(900.0, 1100.0));
        for _ in 0..40 {
            w.update(1.0 / 60.0);
        }
        let burned = w
            .ents
            .iter()
            .filter(|e| e.team == Team::Enemy && e.kind == Kind::Soldier && e.pos.x < 700.0 && e.hp < e.max_hp)
            .count();
        assert!(burned >= 2, "flame cone should hit the whole cluster ({burned})");
        let far_e = w.ents.iter().find(|e| e.id == w.ents[far].id).unwrap();
        assert_eq!(far_e.hp, far_e.max_hp, "the distant unit should be out of flame range");
    }

    #[test]
    fn mortar_shells_from_range_with_a_point_blank_dead_zone() {
        // In its firing band, a mortar shells a target. (No AI, so the dummy holds still.)
        let mut w = World::new_match(2, 2, [false, false, false, false], Team::Player, true);
        w.flatten_terrain();
        w.spawn(Kind::Mortar, Team::Player, v2(900.0, 900.0));
        let far = w.spawn(Kind::Worker, Team::Enemy, v2(1050.0, 900.0)); // 150px, in band
        let fid = w.ents[far].id;
        let far_hp = w.ents[far].hp;
        for _ in 0..(60 * 2) {
            w.update(1.0 / 60.0);
        }
        let hit = w.index_of(fid).map_or(true, |j| w.ents[j].hp < far_hp);
        assert!(hit, "mortar should shell a target in its firing band");

        // Point-blank: a target inside the dead zone takes no fire before the
        // mortar has a chance to kite back out.
        let mut w2 = World::new_match(2, 2, [false, false, false, false], Team::Player, true);
        w2.flatten_terrain();
        w2.spawn(Kind::Mortar, Team::Player, v2(900.0, 900.0));
        let near = w2.spawn(Kind::Worker, Team::Enemy, v2(940.0, 900.0)); // 40px < 85 dead zone
        let nid = w2.ents[near].id;
        let near_hp = w2.ents[near].hp;
        for _ in 0..12 {
            w2.update(1.0 / 60.0);
        }
        assert_eq!(
            w2.ents[w2.index_of(nid).unwrap()].hp,
            near_hp,
            "mortar must not fire into its point-blank dead zone"
        );
    }

    #[test]
    fn sapper_detonates_on_contact_and_dies() {
        let mut w = World::new_match(2, 2, [false, false, false, false], Team::Player, true);
        w.flatten_terrain();
        let s = w.spawn(Kind::Sapper, Team::Player, v2(900.0, 900.0));
        let sid = w.ents[s].id;
        let mut eids = Vec::new();
        for off in [v2(20.0, 0.0), v2(34.0, 12.0), v2(8.0, 30.0)] {
            let e = w.spawn(Kind::Soldier, Team::Enemy, v2(900.0, 900.0).add(off));
            eids.push((w.ents[e].id, w.ents[e].hp));
        }
        for _ in 0..30 {
            w.update(1.0 / 60.0);
        }
        assert!(w.index_of(sid).is_none(), "the sapper should die in its own blast");
        let hurt = eids
            .iter()
            .any(|&(id, hp0)| w.index_of(id).map_or(true, |j| w.ents[j].hp < hp0));
        assert!(hurt, "the blast should damage the clustered enemies");
    }

    #[test]
    fn minimap_attack_snaps_to_enemy() {
        let mut w = World::new(7);
        let s = w.spawn(Kind::Soldier, Team::Player, v2(400.0, 1100.0));
        w.ents[s].selected = true;
        let e = w.spawn(Kind::Soldier, Team::Enemy, v2(1500.0, 400.0));
        let (epos, eid) = (w.ents[e].pos, w.ents[e].id);
        // An imprecise click near the enemy (as you'd get on the minimap).
        let clicked = epos.add(v2(60.0, 60.0));
        let snapped = w.snap_to_enemy(clicked, 130.0).expect("should snap onto the enemy");
        assert!(snapped.dist(epos) < 1.0);
        w.command(snapped, false);
        assert!(
            matches!(w.ents[s].order, Order::Attack(t) if t == eid),
            "minimap click near an enemy should issue an attack"
        );
    }

    #[test]
    fn move_command_overrides_combat() {
        let mut w = World::new(7);
        w.flatten_terrain(); // tests command-override logic, not terrain routing
        let si = w.spawn(Kind::Soldier, Team::Player, v2(500.0, 500.0));
        let sid = w.ents[si].id;
        w.spawn(Kind::Soldier, Team::Enemy, v2(560.0, 500.0)); // right next to it
        w.ents[si].selected = true;
        w.update(1.0 / 60.0); // it auto-engages the adjacent enemy
        // Order it to move far south, away from the fight.
        w.command(v2(500.0, 1300.0), false);
        let y0 = w.ents[w.index_of(sid).unwrap()].pos.y;
        for _ in 0..90 {
            w.update(1.0 / 60.0);
        }
        let moved = w.index_of(sid).map_or(false, |i| w.ents[i].pos.y > y0 + 60.0);
        assert!(moved, "a Move order must override an active attack");
    }

    #[test]
    fn right_click_cancels_queued_unit() {
        let mut w = World::new(7);
        let b = w.spawn(Kind::Barracks, Team::Player, v2(500.0, 1100.0));
        w.ents[b].selected = true;
        w.minerals[0] = 1000;
        assert!(w.try_train(b, Kind::Soldier));
        assert!(w.try_train(b, Kind::Soldier));
        assert_eq!(w.ents[b].queue.len(), 2);
        let min_before = w.minerals[0];
        let bpos = w.ents[b].pos;
        w.command(bpos, false); // right-click the barracks
        assert_eq!(w.ents[b].queue.len(), 1, "one queued unit should be cancelled");
        assert_eq!(
            w.minerals[0],
            min_before + cost(Kind::Soldier),
            "the cost should be refunded"
        );
    }

    #[test]
    fn fog_reveals_own_area_hides_enemy() {
        let mut w = World::new(7);
        w.update(1.0 / 60.0); // compute initial visibility
        let ppos = w.ents[w.first_base(Team::Player).unwrap()].pos;
        let epos = w.ents[w.first_base(Team::Enemy).unwrap()].pos;
        assert_eq!(w.vis_at(ppos), 2, "your own base should be visible");
        assert_eq!(w.vis_at(epos), 0, "the enemy base should start hidden in fog");
    }

    #[test]
    fn death_emits_particles() {
        let mut w = World::new(7);
        let id = w.spawn(Kind::Soldier, Team::Enemy, v2(500.0, 500.0));
        w.ents[id].hp = -1.0; // mark dead
        assert!(w.particles.is_empty());
        w.update(1.0 / 60.0); // cleanup() runs the death effect
        assert!(!w.particles.is_empty(), "a death should throw particles");
    }

    #[test]
    fn economy_actually_accrues() {
        // Workers mine and deposit: player minerals should climb while idle.
        let mut w = World::new(1);
        let start = w.minerals[0];
        for _ in 0..(60 * 30) {
            w.update(1.0 / 60.0);
        }
        assert!(w.minerals[0] > start, "auto-mining workers should earn minerals");
    }

    #[test]
    fn workers_mine_at_full_throughput() {
        // Regression guard: a worker's drop-off goal is its base centre, which
        // sits on a tile the path planner marks solid (the building footprint).
        // If pathing isn't short-circuited near the goal, workers circle their
        // own base instead of depositing and throughput collapses (~48/30s).
        // Healthy starting economy is several hundred minerals over 30s.
        let mut w = World::new(1);
        let start = w.minerals[0];
        for _ in 0..(60 * 30) {
            w.update(1.0 / 60.0);
        }
        let gained = w.minerals[0] - start;
        assert!(
            gained >= 250,
            "starting workers should mine freely, only gained {gained} in 30s"
        );
    }

    #[test]
    fn starting_workers_fan_out_across_patches() {
        // The crew should spread over the home mineral line, not dog-pile the
        // single closest patch.
        let w = World::new(1);
        let mut targets: Vec<u32> = w
            .ents
            .iter()
            .filter(|e| e.kind == Kind::Worker && e.team == Team::Player)
            .filter_map(|e| match e.order {
                Order::Gather(id) => Some(id),
                _ => None,
            })
            .collect();
        let workers = targets.len();
        targets.sort_unstable();
        targets.dedup();
        assert_eq!(workers, 4, "four starting workers expected");
        assert!(
            targets.len() >= 3,
            "workers should fan out, but only {} distinct patches targeted",
            targets.len()
        );
    }

    #[test]
    fn ctrl_a_selects_army_but_not_workers() {
        let mut w = World::new(2);
        let bp = w.ents[w.first_base(Team::Player).unwrap()].pos;
        let s = w.spawn(Kind::Soldier, Team::Player, bp.add(v2(60.0, 0.0)));
        let t = w.spawn(Kind::Tank, Team::Player, bp.add(v2(90.0, 0.0)));
        w.select_all_army(false);
        assert!(w.ents[s].selected, "soldiers should join the army selection");
        assert!(w.ents[t].selected, "tanks should join the army selection");
        assert!(
            w.ents.iter().all(|e| !(e.selected && e.kind == Kind::Worker)),
            "workers must be excluded from Ctrl+A"
        );
        assert!(
            w.ents
                .iter()
                .all(|e| !(e.selected && e.team != Team::Player)),
            "Ctrl+A only grabs your own units"
        );
    }
}

