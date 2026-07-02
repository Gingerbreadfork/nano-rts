//! The opponent. A *fog-limited* brain: it only knows what its own units can
//! see, scouts to learn more, and picks from a grab-bag of intents — build,
//! harass, feint, commit, defend — on an irregular, dice-driven timer.
//!
//! One brain runs per AI faction (`ai_update(w, dt, me)`); every other living
//! faction is an enemy, so the same code drives 1v1 and free-for-alls. Each
//! faction's `AiState` (personality + intel) lives in `w.ai[me.idx()]`.

use crate::vec::{v2, V2};
use crate::world::{cost, is_army, is_building, supply_cost, tie_key, Intent, Kind, Order, Strategy, Team, World};

/// Is `e` an enemy of faction `me` (a different, non-neutral faction)?
#[inline]
fn is_foe(team: Team, me: Team) -> bool {
    team != me && team != Team::Neutral
}

fn map_center(w: &World) -> V2 {
    v2(w.world_w * 0.5, w.world_h * 0.5)
}

fn nearest_enemy_base(w: &World, me: Team, from: V2) -> Option<V2> {
    let mut best = None;
    let mut bd = f32::MAX;
    let mut bk = 0u64;
    for e in &w.ents {
        if e.kind == Kind::Base && is_foe(e.team, me) {
            let d = e.pos.dist_sq(from);
            // Equidistant rival bases: pick by a hashed id, not list order, so
            // no faction (least of all the human) is everyone's default target.
            let k = tie_key(e.id);
            if best.is_none() || d < bd || (d == bd && k < bk) {
                bd = d;
                bk = k;
                best = Some(e.pos);
            }
        }
    }
    best
}

pub fn ai_update(w: &mut World, dt: f32, me: Team) {
    let mi = me.idx();
    // Intent / scout / intel timers tick in real time, between decisions too.
    w.ai[mi].intent_timer -= dt;
    w.ai[mi].scout_timer -= dt;
    w.ai[mi].seen_army_age += dt;

    w.ai[mi].think -= dt;
    if w.ai[mi].think > 0.0 {
        return;
    }
    w.ai[mi].think = 0.35;

    // Anchor on our Command Center if we have one; otherwise on any surviving
    // building, so losing the CC doesn't lobotomise the brain — it keeps fighting
    // from its barracks/factory and rebuilds a CC when it can afford one.
    let have_base = w.first_base(me).is_some();
    let anchor = w
        .first_base(me)
        .or_else(|| w.ents.iter().position(|e| e.team == me && is_building(e.kind)));
    let Some(anchor_i) = anchor else {
        return; // no buildings left — eliminated
    };
    let base_pos = w.ents[anchor_i].pos;
    // Keep the primary objective pointed at the nearest living enemy base.
    if let Some(p) = nearest_enemy_base(w, me, base_pos) {
        w.ai[mi].player_main = p;
    }
    let staging = w.ai[mi].staging;
    let strat = w.ai[mi].strategy;
    let scout_id = w.ai[mi].scout_id;
    // Building sites cluster between the base and the map centre (works from any
    // corner), with `perp` spreading them sideways.
    let toward = map_center(w).sub(base_pos).norm();
    let perp = v2(-toward.y, toward.x);
    let site = |fwd: f32, side: f32| base_pos.add(toward.scale(fwd)).add(perp.scale(side));

    // ---------------- census of our own forces ----------------
    let mut workers = 0u32;
    let mut idle_workers: Vec<usize> = Vec::new();
    let mut barracks: Vec<usize> = Vec::new();
    let mut factories: Vec<usize> = Vec::new();
    let mut bases: Vec<usize> = Vec::new();
    let mut army: Vec<usize> = Vec::new(); // soldiers + tanks, minus the scout
    let mut army_supply = 0u32;

    for (i, e) in w.ents.iter().enumerate() {
        if e.team != me {
            continue;
        }
        match e.kind {
            Kind::Worker => {
                workers += 1;
                if matches!(e.order, Order::Idle) {
                    idle_workers.push(i);
                }
            }
            k if is_army(k) => {
                if e.id != scout_id {
                    army.push(i);
                    army_supply += supply_cost(e.kind);
                }
            }
            Kind::Barracks if e.build_left <= 0.0 => barracks.push(i),
            Kind::Factory if e.build_left <= 0.0 => factories.push(i),
            Kind::Base => bases.push(i),
            _ => {}
        }
    }

    let minerals = w.team_min(me);
    let used = w.supply_used(me);
    let cap = w.supply_cap(me);

    // ---------------- economy ----------------
    for &wi in &idle_workers {
        if w.ents[wi].id == scout_id {
            continue;
        }
        if let Some(m) = w.nearest_mineral_idx(w.ents[wi].pos) {
            let id = w.ents[m].id;
            w.ents[wi].order = Order::Gather(id);
        }
    }

    // Lost the Command Center but still standing? Rebuild one as the top priority
    // so workers can be trained again — a real comeback instead of withering.
    if !have_base && minerals >= cost(Kind::Base) && !pending(w, Kind::Base, me) {
        if let (Some(wi), Some(s)) = (pick_worker(w, me), free_site(w, base_pos, Kind::Base, me)) {
            w.order_build(wi, Kind::Base, s);
        }
    }

    let worker_target = (w.ai[mi].worker_target + bases.len().saturating_sub(1) as u32 * 8).min(26);
    if workers < worker_target && minerals >= cost(Kind::Worker) {
        for &bi in &bases {
            if w.ents[bi].queue.is_empty() && w.ents[bi].build_left <= 0.0 && w.try_train(bi, Kind::Worker) {
                break;
            }
        }
    }

    if cap < 120 && (cap as i32 - used as i32) < 6 && minerals >= cost(Kind::Depot) && !pending(w, Kind::Depot, me) {
        if let (Some(wi), Some(s)) = (pick_worker(w, me), free_site(w, site(70.0, 150.0), Kind::Depot, me)) {
            w.order_build(wi, Kind::Depot, s);
        }
    }

    // Repairs: patch up battle damage — urgently below 60% hp, and even light
    // scratches when the bank is fat. Only finished buildings qualify (raising
    // a shell is the builder's job); one worker per building, worst-hurt first.
    let threshold = if minerals > 300 { 0.95 } else { 0.6 };
    let mut fix: Option<usize> = None;
    let mut worst = threshold;
    for (i, e) in w.ents.iter().enumerate() {
        if e.team != me || !is_building(e.kind) || e.build_left > 0.0 {
            continue;
        }
        let frac = e.hp / e.max_hp;
        if frac >= worst {
            continue; // strict `<` keeps the earlier (lowest-id) tie
        }
        let bid = e.id;
        if w.ents.iter().any(|u| u.team == me && matches!(u.order, Order::Repair(r) if r == bid)) {
            continue; // already has a mechanic on the way
        }
        worst = frac;
        fix = Some(i);
    }
    if let Some(bi) = fix {
        let (bid, bp) = (w.ents[bi].id, w.ents[bi].pos);
        // The nearest free worker: not the scout, not a builder (redirecting one
        // would strand its reserved minerals), not already repairing something.
        let mut pick: Option<usize> = None;
        let mut pd = f32::MAX;
        for (i, e) in w.ents.iter().enumerate() {
            if e.team != me || e.kind != Kind::Worker || e.id == scout_id {
                continue;
            }
            if matches!(e.order, Order::Build(_, _) | Order::Repair(_)) || !e.build_queue.is_empty() {
                continue;
            }
            let d = e.pos.dist_sq(bp);
            if d < pd {
                pd = d;
                pick = Some(i);
            }
        }
        if let Some(wi) = pick {
            w.ents[wi].order = Order::Repair(bid);
        }
    }

    // ---------------- tech (timing varies by personality) ----------------
    if barracks.is_empty() && !pending(w, Kind::Barracks, me) && minerals >= cost(Kind::Barracks) {
        if let (Some(wi), Some(s)) = (pick_worker(w, me), free_site(w, site(120.0, 60.0), Kind::Barracks, me)) {
            w.order_build(wi, Kind::Barracks, s);
        }
    }

    let factory_ready = match strat {
        Strategy::Mech => workers >= 6,
        Strategy::Rush => army_supply >= 4 && workers >= 8,
        _ => workers >= 8,
    };
    if factories.is_empty() && !pending(w, Kind::Factory, me) && !barracks.is_empty() && factory_ready && minerals >= cost(Kind::Factory) {
        if let (Some(wi), Some(s)) = (pick_worker(w, me), free_site(w, site(150.0, -40.0), Kind::Factory, me)) {
            w.order_build(wi, Kind::Factory, s);
        }
    }

    // A second production building once established, biased by personality.
    if w.time > 80.0 && minerals >= cost(Kind::Barracks) + 80 {
        let prefer_factory = strat == Strategy::Mech || w.ai[mi].tank_ratio > 0.5;
        if prefer_factory && !factories.is_empty() && factories.len() < 2 && !pending(w, Kind::Factory, me) {
            if let (Some(wi), Some(s)) = (pick_worker(w, me), free_site(w, site(120.0, -120.0), Kind::Factory, me)) {
                w.order_build(wi, Kind::Factory, s);
            }
        } else if barracks.len() < 2 && !pending(w, Kind::Barracks, me) {
            if let (Some(wi), Some(s)) = (pick_worker(w, me), free_site(w, site(80.0, 200.0), Kind::Barracks, me)) {
                w.order_build(wi, Kind::Barracks, s);
            }
        }
    }

    // ---------------- expansion (threshold varies) ----------------
    // No one-shot latch here: the base census counts a rising expansion and
    // `pending` covers the builder walking out, so if the builder is sniped or
    // the expansion razed, this gate simply opens again and the AI retries.
    if bases.len() == 1
        && workers >= 10
        && !barracks.is_empty()
        && minerals >= w.ai[mi].expand_min
        && !pending(w, Kind::Base, me)
    {
        if let (Some(wi), Some(s)) = (pick_worker(w, me), expansion_site(w, base_pos, me)) {
            w.order_build(wi, Kind::Base, s);
        }
    }

    // ---------------- army production (mix follows tank_ratio) ----------------
    for &bi in barracks.iter().chain(factories.iter()) {
        w.ents[bi].rally = staging;
    }
    let tanks_first = w.rng_f() < w.ai[mi].tank_ratio;
    if tanks_first {
        train_tanks(w, &factories, mi);
        train_soldiers(w, &barracks, mi);
    } else {
        train_soldiers(w, &barracks, mi);
        train_tanks(w, &factories, mi);
    }

    // ---------------- intel, scouting, doctrine ----------------
    update_intel(w, me);
    maybe_scout(w, &army, &idle_workers, workers, me);
    run_doctrine(w, &army, army_supply, &bases, staging, me);
}

fn train_soldiers(w: &mut World, barracks: &[usize], mi: usize) {
    for &bi in barracks {
        if w.ents[bi].queue.len() < 2 {
            let r = w.rng_f();
            let (sap, pyro) = (w.ai[mi].sapper_ratio, w.ai[mi].pyro_ratio);
            let unit = if r < sap {
                Kind::Sapper
            } else if r < sap + pyro {
                Kind::Pyro
            } else {
                Kind::Soldier
            };
            let _ = w.try_train(bi, unit);
        }
    }
}
fn train_tanks(w: &mut World, factories: &[usize], mi: usize) {
    for &fi in factories {
        if w.ents[fi].queue.len() < 2 {
            let r = w.rng_f();
            let (mortar, raid) = (w.ai[mi].mortar_ratio, w.ai[mi].raider_ratio);
            let unit = if r < mortar {
                Kind::Mortar
            } else if r < mortar + raid {
                Kind::Raider
            } else {
                Kind::Tank
            };
            let _ = w.try_train(fi, unit);
        }
    }
}

// ---------------- intel (fog of war) ---------------------------------------

/// Refresh what faction `me` *knows* from its own fog: where an enemy army was
/// last seen and how big, plus a memory of enemy buildings it has scouted.
fn update_intel(w: &mut World, me: Team) {
    let mi = me.idx();
    let mut sum = v2(0.0, 0.0);
    let mut n = 0u32;
    let mut sup = 0u32;
    let mut new_buildings: Vec<(V2, Kind)> = Vec::new();
    for e in &w.ents {
        if !is_foe(e.team, me) || w.team_vis_at(me, e.pos) != 2 {
            continue;
        }
        match e.kind {
            k if is_army(k) => {
                sum = sum.add(e.pos);
                n += 1;
                sup += supply_cost(e.kind);
            }
            k if is_building(k) => new_buildings.push((e.pos, k)),
            _ => {}
        }
    }
    if n > 0 {
        w.ai[mi].seen_army_pos = sum.scale(1.0 / n as f32);
        w.ai[mi].seen_army_supply = sup;
        w.ai[mi].seen_army_age = 0.0;
    }
    for (p, k) in new_buildings {
        if !w.ai[mi].known.iter().any(|(q, _)| q.dist_sq(p) < 60.0 * 60.0) {
            w.ai[mi].known.push((p, k));
            if w.ai[mi].known.len() > 16 {
                w.ai[mi].known.remove(0);
            }
        }
    }
    // Forget raid targets we can *see* are gone: a remembered spot that is in
    // full view with no enemy building standing on it any more. Otherwise
    // harass and commit keep marching the army onto bare ground.
    let mut k = 0;
    while k < w.ai[mi].known.len() {
        let p = w.ai[mi].known[k].0;
        let visible = w.team_vis_at(me, p) == 2;
        let standing = visible
            && w.ents.iter().any(|e| {
                is_foe(e.team, me) && is_building(e.kind) && e.pos.dist_sq(p) < 60.0 * 60.0
            });
        if visible && !standing {
            w.ai[mi].known.remove(k);
        } else {
            k += 1;
        }
    }
}

// ---------------- scouting -------------------------------------------------

fn maybe_scout(w: &mut World, army: &[usize], idle_workers: &[usize], workers: u32, me: Team) {
    let mi = me.idx();
    if w.ai[mi].scout_id != 0 {
        match w.index_of(w.ai[mi].scout_id) {
            Some(si) => {
                // Reclaim an early *worker* scout once soldiers can take over.
                if w.ents[si].kind == Kind::Worker && !army.is_empty() {
                    w.ents[si].order = Order::Idle;
                    w.ai[mi].scout_id = 0;
                    w.ai[mi].scout_timer = 1.0;
                    return;
                }
                if matches!(w.ents[si].order, Order::Idle) {
                    let wp = recon_point(w, me);
                    w.ents[si].order = Order::Move(wp);
                }
                return;
            }
            None => {
                w.ai[mi].scout_id = 0;
                w.ai[mi].scout_timer = 18.0 + w.rng_f() * 18.0;
            }
        }
    }
    if w.ai[mi].scout_timer > 0.0 {
        return;
    }
    let pick = if !army.is_empty() && w.rng_f() < 0.7 {
        // Prefer fast, expendable eyes: a Raider outruns anything, then plain
        // infantry. A Mortar (slowest unit on the map) or a Sapper (a one-shot
        // bomb wasted on sightseeing) only goes when there is nothing else.
        army.iter()
            .copied()
            .find(|&i| w.ents[i].kind == Kind::Raider)
            .or_else(|| {
                army.iter().copied().find(|&i| matches!(w.ents[i].kind, Kind::Soldier | Kind::Pyro))
            })
            .or_else(|| {
                army.iter().copied().find(|&i| !matches!(w.ents[i].kind, Kind::Mortar | Kind::Sapper))
            })
            .or_else(|| army.last().copied())
    } else if workers > 5 {
        idle_workers.first().copied().or_else(|| {
            w.ents.iter().position(|e| e.team == me && e.kind == Kind::Worker && !matches!(e.order, Order::Build(_, _)))
        })
    } else {
        None
    };
    if let Some(si) = pick {
        w.ai[mi].scout_id = w.ents[si].id;
        let wp = recon_point(w, me);
        w.ents[si].order = Order::Move(wp);
        w.ai[mi].scout_timer = 25.0 + w.rng_f() * 20.0;
    } else {
        w.ai[mi].scout_timer = 4.0;
    }
}

/// A recon waypoint — biased toward the nearest enemy main, with detours through
/// the contested centre and the expansion lane, plus jitter.
fn recon_point(w: &mut World, me: Team) -> V2 {
    let mi = me.idx();
    let main = w.ai[mi].player_main;
    let center = map_center(w);
    let r = w.rng_f();
    let p = if r < 0.5 {
        main.add(v2((w.rng_f() - 0.5) * 320.0, (w.rng_f() - 0.5) * 320.0))
    } else if r < 0.8 {
        center
    } else {
        main.add(center.sub(main).scale(0.5))
    };
    clamp_pt(p, w)
}

// ---------------- doctrine -------------------------------------------------

fn run_doctrine(w: &mut World, army: &[usize], army_supply: u32, bases: &[usize], staging: V2, me: Team) {
    let mi = me.idx();
    // 1) DEFEND — react to *visible* enemies near a base, scaled to how big the
    // raid actually is. Workers count as attackers: they only hit for 3, but a
    // worker rush in numbers is a real threat.
    let mut threat: Option<V2> = None; // the attacker nearest to any base
    let mut home = base_fallback(w, bases, staging); // the base under threat
    let mut threat_supply = 0u32;
    let mut attackers: Vec<usize> = Vec::new();
    let mut bd = 360.0f32 * 360.0;
    for (i, e) in w.ents.iter().enumerate() {
        if !is_foe(e.team, me)
            || !(is_army(e.kind) || e.kind == Kind::Worker)
            || w.team_vis_at(me, e.pos) != 2
        {
            continue;
        }
        let mut nd = f32::MAX;
        let mut nb = home;
        for &bi in bases {
            let d = e.pos.dist_sq(w.ents[bi].pos);
            if d < nd {
                nd = d;
                nb = w.ents[bi].pos;
            }
        }
        if nd >= 360.0 * 360.0 {
            continue;
        }
        attackers.push(i);
        threat_supply += supply_cost(e.kind);
        if nd < bd {
            bd = nd;
            threat = Some(e.pos);
            home = nb;
        }
    }
    if let Some(tp) = threat {
        // With the army clearly outmatched (or away), the mineral line fights
        // too — workers have no aggro, so they need explicit Attack orders.
        if army_supply < threat_supply {
            defend_with_workers(w, &attackers, home, me);
        }
        if threat_supply >= 4 {
            // A real attack: collapse the whole army onto it, preempting any plan.
            command_army(w, army, tp, mi);
            w.ai[mi].intent = Intent::Defend(tp);
            w.ai[mi].intent_timer = w.ai[mi].intent_timer.max(1.5);
            return;
        }
        // One or two stray units: peel a small response instead of letting a
        // lone parked scout yo-yo the entire army home every think tick.
        let peel = peel_defenders(w, army, tp, mi);
        let rest: Vec<usize> = army.iter().copied().filter(|i| !peel.contains(i)).collect();
        execute_intent(w, &rest, army_supply, staging, me);
        return;
    }

    execute_intent(w, army, army_supply, staging, me);
}

/// The threatened-base anchor when no base is closer: the first base, or the
/// staging point if (somehow) none stands.
fn base_fallback(w: &World, bases: &[usize], staging: V2) -> V2 {
    bases.first().map(|&bi| w.ents[bi].pos).unwrap_or(staging)
}

/// Send the 2-3 army units nearest the threat at it, leaving the rest to their
/// current doctrine. Deterministic: sorted by distance, ties by entity index.
fn peel_defenders(w: &mut World, army: &[usize], tp: V2, mi: usize) -> Vec<usize> {
    let scout = w.ai[mi].scout_id;
    let mut ranked: Vec<(f32, usize)> = army
        .iter()
        .copied()
        .filter(|&i| w.ents[i].id != scout)
        .map(|i| (w.ents[i].pos.dist_sq(tp), i))
        .collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    ranked.truncate(3);
    let pt = clamp_pt(tp, w);
    let peel: Vec<usize> = ranked.iter().map(|&(_, i)| i).collect();
    for &i in &peel {
        w.ents[i].order = Order::AttackMove(pt);
    }
    peel
}

/// Emergency worker defense. AttackMove is useless here — workers have zero
/// aggro and would just stand around — so each gets an explicit Attack on the
/// raider nearest to it. Pull at most twice the attacker count from around the
/// threatened base; when the threat dies they go Idle and the economy loop
/// re-gathers them. Deterministic: ents order, strict `<` nearest-target.
fn defend_with_workers(w: &mut World, attackers: &[usize], home: V2, me: Team) {
    let scout = w.ai[me.idx()].scout_id;
    let want = attackers.len() * 2;
    let mut picks: Vec<usize> = Vec::new();
    for (i, e) in w.ents.iter().enumerate() {
        if picks.len() >= want {
            break;
        }
        if e.team != me || e.kind != Kind::Worker || e.id == scout {
            continue;
        }
        // Builders keep building: redirecting one would strand reserved minerals.
        if matches!(e.order, Order::Build(_, _)) || !e.build_queue.is_empty() {
            continue;
        }
        if e.pos.dist_sq(home) > 400.0 * 400.0 {
            continue;
        }
        picks.push(i);
    }
    for &wi in &picks {
        let wp = w.ents[wi].pos;
        let mut tgt: Option<u32> = None;
        let mut td = f32::MAX;
        for &ai in attackers {
            let d = w.ents[ai].pos.dist_sq(wp);
            if d < td {
                td = d;
                tgt = Some(w.ents[ai].id);
            }
        }
        if let Some(id) = tgt {
            w.ents[wi].order = Order::Attack(id);
        }
    }
}

/// 2) Re-evaluate the intent when its timer lapses, then push the army along it.
fn execute_intent(w: &mut World, army: &[usize], army_supply: u32, staging: V2, me: Team) {
    let mi = me.idx();
    if w.ai[mi].intent_timer <= 0.0 || matches!(w.ai[mi].intent, Intent::Defend(_)) {
        choose_intent(w, army_supply, me);
    }

    match w.ai[mi].intent {
        Intent::Build => command_army(w, army, staging, mi),
        Intent::Harass(tp) => {
            let squad = (army.len() / 3).clamp(1, 4);
            let scout = w.ai[mi].scout_id;
            for (k, &i) in army.iter().enumerate() {
                if w.ents[i].id == scout {
                    continue;
                }
                let dst = if k < squad { tp } else { staging };
                w.ents[i].order = Order::AttackMove(dst);
            }
        }
        Intent::Commit(tp) => {
            command_army(w, army, tp, mi);
            let frac = 0.30 + 0.25 * (1.0 - w.ai[mi].aggression);
            if (army_supply as f32) < w.ai[mi].commit_army as f32 * frac {
                w.ai[mi].intent = Intent::Build;
                w.ai[mi].intent_timer = 4.0 + w.rng_f() * 3.0;
            }
        }
        Intent::Feint(tp) => {
            let dst = if w.ai[mi].intent_timer < 1.4 { staging } else { tp };
            command_army(w, army, dst, mi);
        }
        Intent::Defend(tp) => command_army(w, army, tp, mi),
    }
}

fn command_army(w: &mut World, army: &[usize], dst: V2, mi: usize) {
    let scout = w.ai[mi].scout_id;
    // Army centre and the heading toward the objective, so each unit can be placed
    // relative to the front: the fragile siege/flame line tucked behind the
    // bruisers, raiders swinging wide to flank instead of feeding the deathball.
    let mut c = v2(0.0, 0.0);
    let mut n = 0.0f32;
    for &i in army {
        if w.ents[i].id == scout {
            continue;
        }
        c = c.add(w.ents[i].pos);
        n += 1.0;
    }
    if n < 1.0 {
        return;
    }
    c = c.scale(1.0 / n);
    let fwd = dst.sub(c);
    let fwd = if fwd.len() > 1.0 { fwd.norm() } else { v2(0.0, 1.0) };
    let perp = v2(-fwd.y, fwd.x);
    for &i in army {
        if w.ents[i].id == scout {
            continue;
        }
        let raw = match w.ents[i].kind {
            // Mortars lob from the back line — never charge into their dead zone.
            Kind::Mortar => dst.sub(fwd.scale(150.0)),
            // Pyros are short-ranged and fragile: tuck in just behind the front.
            Kind::Pyro => dst.sub(fwd.scale(55.0)),
            // Raiders swing wide to flank the soft backline.
            Kind::Raider => dst.add(perp.scale(if w.ents[i].id % 2 == 0 { 150.0 } else { -150.0 })),
            // Soldiers, Tanks, Sappers hold the line and crash straight in.
            _ => dst,
        };
        let pt = clamp_pt(raw, w);
        w.ents[i].order = Order::AttackMove(pt);
    }
}

fn choose_intent(w: &mut World, army_supply: u32, me: Team) {
    let mi = me.idx();
    let agg = w.ai[mi].aggression;
    let est = if w.ai[mi].seen_army_age < 10.0 {
        w.ai[mi].seen_army_supply
    } else {
        // Stale intel decays: shave ~1 supply per 3 seconds of age before the
        // uncertainty bump, so an opponent whose army died long ago eventually
        // reads as weak instead of freezing the AI at home against a ghost.
        let decayed = (w.ai[mi].seen_army_supply as f32 - w.ai[mi].seen_army_age / 3.0).max(0.0);
        (decayed * 1.4) as u32 + 4
    };
    let needed = ((est as f32 * (1.35 - 0.7 * agg)) as u32).max(4);

    let r = w.rng_f();
    if army_supply >= needed && army_supply >= 3 {
        if r < 0.16 * (1.0 - agg) {
            let tp = approach(w, false, me);
            w.ai[mi].intent = Intent::Feint(tp);
            w.ai[mi].intent_timer = 2.5 + w.rng_f() * 2.0;
        } else {
            let tp = commit_target(w, me);
            w.ai[mi].intent = Intent::Commit(tp);
            w.ai[mi].commit_army = army_supply;
            w.ai[mi].intent_timer = 5.0 + w.ai[mi].patience + w.rng_f() * 3.0;
        }
    } else {
        let poke_chance = if w.ai[mi].harass { 0.35 } else { 0.12 };
        if army_supply >= 3 && r < poke_chance {
            let tp = harass_target(w, me);
            w.ai[mi].intent = Intent::Harass(tp);
            w.ai[mi].intent_timer = 3.5 + w.rng_f() * 3.0;
        } else {
            w.ai[mi].intent = Intent::Build;
            w.ai[mi].intent_timer = 2.0 + w.rng_f() * 2.5;
        }
    }
}

fn commit_target(w: &mut World, me: Team) -> V2 {
    let mi = me.idx();
    let r = w.rng_f();
    if r < 0.55 {
        let flank = w.rng_f() < 0.4;
        approach(w, flank, me)
    } else if r < 0.75 && w.ai[mi].seen_army_age < 8.0 {
        w.ai[mi].seen_army_pos
    } else if r < 0.9 {
        harass_target(w, me)
    } else {
        approach(w, true, me)
    }
}

fn approach(w: &mut World, flank: bool, me: Team) -> V2 {
    let mi = me.idx();
    let main = w.ai[mi].player_main;
    if !flank {
        return main;
    }
    let dir = main.sub(w.ai[mi].staging).norm();
    let perp = v2(-dir.y, dir.x);
    let side = if w.rng_f() < 0.5 { 1.0 } else { -1.0 };
    clamp_pt(main.add(perp.scale(260.0 * side)), w)
}

fn harass_target(w: &mut World, me: Team) -> V2 {
    let mi = me.idx();
    let main = w.ai[mi].player_main;
    let mut pick: Option<V2> = None;
    for &(p, k) in &w.ai[mi].known {
        if k == Kind::Base && p.dist_sq(main) < 220.0 * 220.0 {
            continue;
        }
        pick = Some(p);
    }
    pick.unwrap_or_else(|| clamp_pt(main.add(v2(60.0, -120.0)), w))
}

// ---------------- shared helpers -------------------------------------------

/// Where to expand: the closest live mineral field that no own base already
/// serves, with the new Command Center hugging the patch on the home side —
/// the same base-beside-the-line geometry the starting positions get. Falls
/// back to a forward base toward the centre if every field is taken.
/// Deterministic: scans ents in order; strict `<` keeps the first (lowest id).
fn expansion_site(w: &World, base_pos: V2, me: Team) -> Option<V2> {
    let mut patch: Option<V2> = None;
    let mut bd = f32::MAX;
    for e in &w.ents {
        if e.kind != Kind::Mineral || e.minerals == 0 {
            continue;
        }
        let taken = w.ents.iter().any(|b| {
            b.team == me && b.kind == Kind::Base && b.pos.dist_sq(e.pos) < 300.0 * 300.0
        });
        if taken {
            continue;
        }
        let d = e.pos.dist_sq(base_pos);
        if d < bd {
            bd = d;
            patch = Some(e.pos);
        }
    }
    let near = match patch {
        Some(p) => p.add(base_pos.sub(p).norm().scale(110.0)),
        None => {
            let toward = map_center(w).sub(base_pos);
            base_pos.add(toward.scale(0.42))
        }
    };
    free_site(w, near, Kind::Base, me)
}

fn pick_worker(w: &World, me: Team) -> Option<usize> {
    let scout = w.ai[me.idx()].scout_id;
    w.ents.iter().position(|e| {
        e.team == me && e.kind == Kind::Worker && e.id != scout && !matches!(e.order, Order::Build(_, _))
    })
}

/// Is a `kind` building already on the way for `me`? Covers the whole pipeline:
/// a worker walking to the site, a chained build it has queued, and the shell
/// itself still rising. The last matters most — the worker is freed the moment
/// the shell spawns, so without it the brain re-orders the same building every
/// think tick for the entire construction window.
fn pending(w: &World, kind: Kind, me: Team) -> bool {
    w.ents.iter().any(|e| {
        e.team == me
            && ((e.kind == kind && e.build_left > 0.0)
                || matches!(e.order, Order::Build(k, _) if k == kind)
                || e.build_queue.iter().any(|&(k, _)| k == kind))
    })
}

fn free_site(w: &World, near: V2, kind: Kind, me: Team) -> Option<V2> {
    // The 8 ring directions (k/8 * TAU) as literal (cos, sin) pairs. Building
    // positions are simulation state (checksummed in lockstep), and libm's f32
    // cos/sin is not bit-identical across platforms — a const table is.
    const RING: [(f32, f32); 8] = [
        (1.0, 0.0),
        (0.70710678, 0.70710678),
        (0.0, 1.0),
        (-0.70710678, 0.70710678),
        (-1.0, 0.0),
        (-0.70710678, -0.70710678),
        (0.0, -1.0),
        (0.70710678, -0.70710678),
    ];
    // The team-scoped check: placement is sim state, so it must see exactly
    // this faction's pending sites — never my_team's, never a rival's.
    if w.can_build_team(me, kind, near) {
        return Some(near);
    }
    for &r in &[64.0f32, 104.0, 150.0, 200.0] {
        for &(c, s) in &RING {
            let p = near.add(v2(c * r, s * r));
            if w.can_build_team(me, kind, p) {
                return Some(p);
            }
        }
    }
    None
}

fn clamp_pt(p: V2, w: &World) -> V2 {
    v2(p.x.clamp(60.0, w.world_w - 60.0), p.y.clamp(60.0, w.world_h - 60.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Step just the enemy brain through several think ticks (0.35s cadence),
    /// without running the rest of the sim.
    fn think(w: &mut World, ticks: u32) {
        for _ in 0..ticks {
            ai_update(w, 0.4, Team::Enemy);
        }
    }

    fn enemy_base_pos(w: &World) -> V2 {
        w.ents[w.first_base(Team::Enemy).unwrap()].pos
    }

    #[test]
    fn no_second_depot_while_one_is_rising() {
        let mut w = World::new(3);
        w.flatten_terrain();
        w.minerals[Team::Enemy.idx()] = 9999;
        let bp = enemy_base_pos(&w);
        // Push the faction up near its supply cap (7 workers against cap 11)...
        for k in 0..3 {
            w.spawn(Kind::Worker, Team::Enemy, bp.add(v2(-60.0 - 20.0 * k as f32, 90.0)));
        }
        // ...with a Depot already rising: its builder has been freed, so only
        // the shell itself marks the build as pending.
        let d = w.spawn(Kind::Depot, Team::Enemy, bp.add(v2(0.0, 150.0)));
        w.ents[d].build_left = 6.0;
        think(&mut w, 10);
        let depots = w
            .ents
            .iter()
            .filter(|e| e.team == Team::Enemy && e.kind == Kind::Depot)
            .count();
        assert_eq!(depots, 1, "the rising depot is the only one");
        assert!(
            !w.ents.iter().any(|e| e.team == Team::Enemy
                && (matches!(e.order, Order::Build(Kind::Depot, _))
                    || e.build_queue.iter().any(|&(k, _)| k == Kind::Depot))),
            "a rising depot counts as pending; no worker should be sent to build another"
        );
    }

    #[test]
    fn expansion_retries_after_a_failed_attempt() {
        let mut w = World::new(3);
        w.flatten_terrain();
        let mi = Team::Enemy.idx();
        // The state a one-shot latch would have poisoned: an expansion was
        // ordered once but the builder died — one base, nothing pending.
        w.ai[mi].expand_min = 400;
        w.minerals[mi] = 2000;
        let bp = enemy_base_pos(&w);
        w.spawn(Kind::Barracks, Team::Enemy, bp.add(v2(0.0, 220.0)));
        for k in 0..8 {
            w.spawn(Kind::Worker, Team::Enemy, bp.add(v2(-60.0 - 20.0 * k as f32, 90.0)));
        }
        think(&mut w, 12);
        assert!(
            w.ents
                .iter()
                .any(|e| e.team == Team::Enemy && matches!(e.order, Order::Build(Kind::Base, _))),
            "the AI should retry the expansion instead of latching off forever"
        );
    }

    #[test]
    fn stale_army_intel_decays_toward_weak() {
        let mut w = World::new(3);
        let mi = Team::Enemy.idx();
        w.ai[mi].aggression = 0.5; // needed == est exactly
        w.ai[mi].seen_army_supply = 40;
        w.ai[mi].seen_army_age = 300.0; // spotted five minutes ago
        choose_intent(&mut w, 6, Team::Enemy);
        assert!(
            matches!(w.ai[mi].intent, Intent::Commit(_) | Intent::Feint(_)),
            "an army last seen minutes ago should no longer scare the AI into turtling"
        );
    }

    #[test]
    fn workers_fight_back_against_a_worker_rush() {
        let mut w = World::new(3);
        w.flatten_terrain();
        let bp = enemy_base_pos(&w);
        // Six player workers storm the enemy base while it has zero army.
        for k in 0..6 {
            w.spawn(Kind::Worker, Team::Player, bp.add(v2(-140.0, -40.0 + 16.0 * k as f32)));
        }
        let mut fought = false;
        for _ in 0..(60 * 5) {
            w.update(1.0 / 60.0);
            if w.ents
                .iter()
                .any(|e| e.team == Team::Enemy && e.kind == Kind::Worker && matches!(e.order, Order::Attack(_)))
            {
                fought = true;
                break;
            }
        }
        assert!(fought, "the mineral line should get explicit Attack orders against a worker rush");
    }
}
