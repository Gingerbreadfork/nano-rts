# nanorts

**RTS in a teacup.** A complete real-time strategy game written from scratch in
Rust with just two dependencies — one to open a window and blit pixels into, one
to reach the speakers.

Simple, yet surprisingly fun.

## The game

Mine minerals, tech up, raise a combined-arms army, and be the **last base
standing**. Play 1v1 or a **2–4 faction free-for-all**: solo against up to 3 AI,
or online with up to 4 humans on separate machines (empty slots fill with AI).
Maps grow with the player count.

**Economy**
- **Workers** mine cyan mineral patches and haul them home (auto-assigned to the
  nearest patch when trained).
- **Supply Depots** raise your population cap — run out of supply and you can't
  train anything until you build more.
- **Expand**: drop a new **Command Center** on a fresh mineral field to out-grow
  your opponent.

**Army** (everything auto-engages enemies it runs into) — six units, each with a
clear job, so the counters matter:
- **Soldiers** (Barracks) — cheap, ranged, the backbone of any army.
- **Pyros** (Barracks) — short-range **flamethrowers** that wash a whole cone in
  fire, melting clumped infantry and workers. Fragile and easily kited, so they
  want to charge in behind a front line.
- **Sappers** (Barracks) — **suicide bombers**: fast and fragile, they charge in
  and detonate for a big blast that shreds clumped armies and buildings. One use,
  high risk — but devastating if they reach the pile.
- **Tanks** (Factory) — slow, heavily armoured, long range, and deal **splash
  damage** that shreds clustered units.
- **Raiders** (Factory) — fast, fragile light vehicles for **harassment**: run
  down workers, snipe expansions, and flank — but they fold to focused fire.
- **Mortars** (Factory) — **siege**: the longest range on the field, lobbing
  splash shells over your front line to break turtles and level buildings. But
  they have a point-blank **dead zone** — get on top of one and it's helpless.

Clump up and a Pyro, Tank, or Sapper punishes you; spread out and you give
Raiders room to pick you apart; turtle up and a Mortar shells you out. Mix your
army to cover its weaknesses — and screen your Mortars so nothing reaches them.

**The map** is generated fresh every match (from the seed, so both peers in
multiplayer get the identical board):
- **High ground** — passable plateaus that grant **extra sight and weapon
  range**. From low ground you *can't see what's on top of a plateau* until
  you're right under it, so holding the high ground baits ambushes.
- **Cliffs** — impassable plateau walls. Units **pathfind around them** (grid
  A\*), so terrain shapes the lanes you fight over.
- **Ramps** — the way up onto high ground.

You can build *on* high ground for a fortified position, but not on cliffs or
ramps. Each faction starts in a corner; bases, mineral lines, and a route
between every pair of starts are always kept clear, so no map is unplayable.

**The opponent** plays a real macro game: it saturates its mineral lines, keeps
ahead of its supply, and techs Barracks → Factory. But it isn't omniscient —
**it has its own fog of war** and only knows what its units can see. It
**scouts** to find your army, and acts on what it last *saw*, so you can hide a
build or a flank in the dark and catch it out.

There's **no wave clock**. Instead it picks an *intent* on an irregular,
dice-driven timer: mass up, **harass** your economy, **commit** the whole army,
**feint** an attack to bait you out, or **collapse home** to defend. Attacks
come from the front, from a flank, or straight at whatever it scouted — and a
rolled personality (aggression, patience, army mix, expansion appetite) means no
two opponents play alike. Reinforcements trickle into the fight continuously
rather than arriving in tidy, predictable waves.

Win by razing every rival's buildings. Losing your Command Center isn't game
over while another building still stands — fight on and rebuild. But once your
last building falls you're out, so the endgame never drags into a hunt for one
fleeing unit.

## Controls

| Input | Action |
|-------|--------|
| Left-drag | Box-select your units |
| Left-click | Select a single unit / building (forgiving hit-test) |
| Double-click | A unit: select every on-screen unit of that type. A building: select the nearby same-kind cluster (train presses then spread evenly across them) |
| Right-click | Move · Attack (enemy) · Gather (mineral) |
| Middle-drag | Grab and pan the map |
| `Ctrl`+`1`–`9` | Assign a control group |
| `1`–`9` | Recall a control group (double-tap to center on it) |
| `Ctrl`+`A` | Select your whole army (every combat unit, no workers) |
| `W` | Train a Worker (Command Center selected) |
| `E` / `Y` / `G` | Train a Soldier / Pyro / Sapper (Barracks selected) |
| `T` / `R` / `V` | Train a Tank / Raider / Mortar (Factory selected) |
| `B` / `F` / `D` / `C` | Build Barracks / Factory / Depot / Command Center (worker selected, then click — **Shift-click to chain several**) |
| `A` | Attack-move (then left-click a destination) |
| `S` | Stop |
| Arrow keys / screen edges | Scroll the map |
| Minimap | Click to jump the camera |
| `H` | Toggle the help overlay |
| `Esc` | Cancel a mode / deselect |
| `R` | Restart (after the game ends) |

## Multiplayer — serverless lockstep

There is **no server**. One player hosts and **up to three others connect** —
the machines play peer-to-peer. The trick: they never send game *state* over the
wire — only *commands*. Every machine runs the exact same deterministic
simulation in **lockstep**, so a few keystrokes per second per player keep up to
four full battles perfectly in sync. (The same property that makes `--sim`
reproducible is what makes this work.)

The host forms a **star**: each player's commands go to the host, which fans
them out to everyone. The host picks one shared seed; commands are scheduled a
few steps ahead (input delay) and a step only runs once a peer holds *every*
human faction's input for it. **AI factions cost no traffic** — they're
simulated identically on every machine. A periodic state checksum is swapped
each way; if any two ever disagree, the HUD raises a **desync** alarm.

Just launch the game and use the **main menu**: **Host Game** opens a lobby
(pick your player count; empty slots become AI), beacons on the LAN, and lists
the players as they join. **Join Game** shows every host it finds on the network
as a one-click button (or type an `IP:PORT` to dial across subnets) — you get a
faction colour and wait for the host to hit **Start**. The host appears
instantly via UDP discovery, so the joiner usually doesn't type anything.

The same flows are available as CLI shortcuts for scripting:

```sh
cargo run --release -- --host          # host (or --host 7777 to pick a port)
cargo run --release -- --join          # auto-discover and join a LAN host
cargo run --release -- --join 192.168.0.10:7777   # join a specific address
```

A live `LOCKSTEP OK · F… · SYNC …` badge shows the shared frame and last
verified-in-sync step. Factions are **blue**, **red**, **green**, **amber** (the
host is always blue).

## Build & run

```sh
cargo run --release
```

The game opens to a menu: **Single Player**, **Host Game**, **Join Game**,
**Quit** — navigate by mouse or the number keys. Pause (`Esc`) and the
end-of-match screen both offer a route back to the menu, so you can play match
after match without relaunching. Add `--windowed` for a normal window instead
of fullscreen.

Runs on **Linux, Windows, and macOS**. It's still a single windowing dependency
(`minifb`) plus `cpal` for audio — no engine, no GPU. The only platform-specific
bit is how it goes fullscreen, and each path is a few `cfg`-gated lines (no extra
crates — just tiny FFI calls):

- **Linux** asks the window manager (`wmctrl`/`xdotool`) to fullscreen a
  borderless window sized via `xrandr`.
- **Windows** opens a borderless window sized to the primary monitor
  (`GetSystemMetrics`) and pins it to the corner.
- **macOS** opens a large titled, resizable window (sized from the main display
  via CoreGraphics) — click the green traffic-light button for the OS's own
  fullscreen; the game rescales live, so it adapts instantly.

`--windowed` (a plain 1600×900 window) always works on every platform.

**Cross-compiling.** Native is easiest (`cargo build --release` on the target OS,
or CI: `windows-latest` / `macos-latest`). To build a Windows `.exe` from Linux
with the MinGW toolchain:

```sh
# Fedora: dnf install mingw64-gcc rust-std-static-x86_64-pc-windows-gnu
# Debian/Ubuntu: apt install mingw-w64 ; rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu   # -> target/.../nanorts.exe
```

Headless determinism / sanity harness (no window required):

```sh
cargo run --release -- --sim
```

Headless lockstep checks (prove peers stay byte-identical):

```sh
# two peers over loopback
cargo run --release -- --mptest-host 7777   # terminal 1
cargo run --release -- --mptest-join 127.0.0.1:7777   # terminal 2

# four human peers in one process (host + 3 joiners through the relay)
cargo run --release -- --mptest4
```

## How it's built

No game engine, no asset files, no GPU. Just a `u32` framebuffer.

| File | Role |
|------|------|
| `src/vec.rs` | The entire math library: a 2D vector. |
| `src/font.rs` | A 5×7 bitmap font, every glyph encoded by hand. |
| `src/gfx.rs` | Software rasterizer — rects, circles, diamonds, lines, text. |
| `src/world.rs` | The simulation: procedural terrain (plateaus/cliffs/ramps) + grid A\* pathfinding, economy, supply, production, combat (incl. tank splash & flame cones), movement with collision avoidance, fog of war, win/lose. |
| `src/ai.rs` | The opponent's brain: macro (saturation, supply, tech, expansion) plus a fog-limited doctrine — scout, read intel, and pick an intent (build / harass / commit / feint / defend) on an irregular timer. |
| `src/audio.rs` | Procedural sound — every shot, boom and chime synthesised live, jittered per trigger; no audio files. |
| `src/net.rs` | Serverless lockstep netcode: command (de)serialization, the TCP transport, the host-relayed N-peer step driver, LAN UDP discovery, and desync detection. |
| `src/main.rs` | Window, input, the app-state loop (menus → match → menu), the front-end (main menu, host/join multiplayer setup), the game loop, and the renderer. |

Units move with direct steering and boid-style separation instead of a full
pathfinder — open terrain keeps it cheap and the bodies never stack. The whole
world is one flat `Vec` of entities ticked at a fixed 60 Hz, which makes the
simulation fully deterministic (see `--sim`).

## License

[MIT](LICENSE).
