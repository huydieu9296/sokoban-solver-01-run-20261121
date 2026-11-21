// Sokoban solver — push-optimal, room by room.
//
// Idea: every level is a hallway with one to four rooms hanging off it, and
// a box that leaves its room can never come back, so the rooms are
// independent puzzles whose push-optima add up to the level's P*. Each room
// is solved on its own with a breadth-first search over pushes (a state is
// the sorted box squares plus the player's reachable region), which is
// push-optimal by construction, and the rooms' move strings are stitched
// together by walking the player between them. Search effort is bounded by
// fixed state counts, never by the clock, and every tie breaks in a fixed
// direction order, so two runs of the program print identical bytes.
//
// Pruning (all of it safe — none of it can throw away a solvable state):
//   * a box on a dead square (no pull path to any goal, other boxes
//     ignored) is stuck for good;
//   * a box inside a frozen 2x2 block of walls and boxes, off its goal,
//     can never move again;
//   * a transposition table keyed by (boxes, region) so no push-state is
//     ever expanded twice.
//
// If a room's plain BFS finds a solution the player cannot walk out of,
// the room is re-solved with the extra goal condition that the player must
// be able to reach the door when the last box lands. If rooms cannot be
// split out at all, the whole level is tried with the same push BFS and
// then with push A* (admissible min-matching heuristic) before giving up
// on that instance with an empty answer.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// State caps. The generator only keeps rooms its own push search solves
// within 250k states, so 300k per room leaves headroom; the run-wide cap
// keeps the whole set far inside the 60 s wall budget even when several
// rooms are hard.
const ROOM_CAP = 300_000;
const RUN_CAP = 3_000_000;
const ASTAR_ROOM_CAP = 600_000;
const WHOLE_CAP = 400_000;

// Directions in a fixed order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  // Whatever the text says, the outer frame is wall.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// Squares a lone box could ever be pushed from onto a goal, found by
// pulling backwards: a box at x pulled one step in direction d came from
// x - d, which needs the player to have stood at x - 2d, so both squares
// must be open floor.
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const from = x - d;
      const behind = x - 2 * d;
      if (from < 0 || behind < 0 || from >= wall.length || behind >= wall.length) continue;
      if (!wall[from] && !wall[behind] && !live[from]) {
        live[from] = 1;
        queue.push(from);
      }
    }
  }
  return live;
}

// Flood-fill the player's reachable region with `occupied` (boxes) in the
// way. Marks every reached square with `stamp` in `seen` and returns the
// region's lowest square, which names the region in a state key: the region
// is fully determined by its lowest square plus the box positions.
function floodRegion(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const up = p - w;
    const down = p + w;
    const left = p - 1;
    const right = p + 1;
    if (!wall[up] && !occupied[up] && seen[up] !== stamp) {
      seen[up] = stamp;
      stack[n++] = up;
    }
    if (!wall[down] && !occupied[down] && seen[down] !== stamp) {
      seen[down] = stamp;
      stack[n++] = down;
    }
    if (!wall[left] && !occupied[left] && seen[left] !== stamp) {
      seen[left] = stamp;
      stack[n++] = left;
    }
    if (!wall[right] && !occupied[right] && seen[right] !== stamp) {
      seen[right] = stamp;
      stack[n++] = right;
    }
  }
  return low;
}

// Is the square the player can reach from `from`, with `occupied` in the
// way? (A flood that only needs a yes/no answer.)
function canReach(
  level: Level,
  occupied: Uint8Array,
  from: number,
  to: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): boolean {
  if (from === to) return true;
  if (level.wall[to] || occupied[to]) return false;
  const { w, wall } = level;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    const up = p - w;
    const down = p + w;
    const left = p - 1;
    const right = p + 1;
    if (!wall[up] && !occupied[up] && seen[up] !== stamp) {
      seen[up] = stamp;
      stack[n++] = up;
    }
    if (!wall[down] && !occupied[down] && seen[down] !== stamp) {
      seen[down] = stamp;
      stack[n++] = down;
    }
    if (!wall[left] && !occupied[left] && seen[left] !== stamp) {
      seen[left] = stamp;
      stack[n++] = left;
    }
    if (!wall[right] && !occupied[right] && seen[right] !== stamp) {
      seen[right] = stamp;
      stack[n++] = right;
    }
  }
  return seen[to] === stamp;
}

// True when the box just pushed onto `t` is caught in a frozen 2x2 block:
// four squares, each a wall or a box, holding at least one box that is off
// its goal. Nothing in such a block can ever move again.
function frozenAt(level: Level, occupied: Uint8Array, t: number): boolean {
  const { w, wall, goalSet } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goalSet[a]) ||
      (occupied[b] && !goalSet[b]) ||
      (occupied[c] && !goalSet[c]) ||
      (occupied[d] && !goalSet[d])
    ) {
      return true;
    }
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes, as lowercase
// step letters. Ties break in the fixed direction order, so it is stable.
function walkTo(
  level: Level,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = STEP[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: target unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

// Where the player ends up after following `moves` from `from`.
function endSquare(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

// State key: the region's lowest square, then the sorted box squares,
// packed as digits in base `size` while that stays exact (at most 4 boxes:
// size ** 5 < 2 ** 53); bigger levels fall back to a string key.
function stateKey(boxes: number[], low: number, size: number): number | string {
  if (boxes.length >= 5) return `${low}|${boxes.join(',')}`;
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

// Admissible push heuristic: the cheapest assignment of boxes to goals on
// Manhattan distance. One push moves one box one Manhattan step closer at
// most, so the assignment sum lower-bounds the pushes still needed.
// Exhaustive over permutations; rooms hold at most 4 boxes.
function matchingHeuristic(level: Level, boxes: number[]): number {
  const { w, goals } = level;
  const n = boxes.length;
  const dist: number[][] = boxes.map((b) => {
    const bx = b % w;
    const by = Math.floor(b / w);
    return goals.map((g) => Math.abs((g % w) - bx) + Math.abs(Math.floor(g / w) - by));
  });
  let best = Infinity;
  const used = new Array<boolean>(goals.length).fill(false);
  const rec = (i: number, sum: number): void => {
    if (sum >= best) return;
    if (i === n) {
      best = sum;
      return;
    }
    for (let j = 0; j < goals.length; j++) {
      if (used[j]) continue;
      used[j] = true;
      rec(i + 1, sum + dist[i]![j]!);
      used[j] = false;
    }
  };
  rec(0, 0);
  return best;
}

interface SearchResult {
  pushes: number;
  moves: string;
}

interface SearchKit {
  level: Level;
  live: Uint8Array;
  occupied: Uint8Array;
  seen: Int32Array;
  mark: Int32Array;
  stack: Int32Array;
  stamp: number;
  budget: { used: number; cap: number };
}

function makeKit(level: Level, budget: { used: number; cap: number }): SearchKit {
  const size = level.wall.length;
  return {
    level,
    live: liveSquares(level),
    occupied: new Uint8Array(size),
    seen: new Int32Array(size),
    mark: new Int32Array(size),
    stack: new Int32Array(size),
    stamp: 0,
    budget,
  };
}

function allOnGoals(level: Level, boxes: number[]): boolean {
  for (const b of boxes) if (!level.goalSet[b]) return false;
  return true;
}

// Rebuild the full move string from a parent chain. For every push on the
// chain, walk the player to the square behind the box, then step through.
function rebuildMoves(
  kit: SearchKit,
  statesBoxes: number[][],
  parent: number[],
  pushFrom: number[],
  pushTo: number[],
  found: number,
): string {
  const { level, occupied } = kit;
  const chain: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of chain) {
    const nextBoxes = statesBoxes[s]!;
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    let dir = -1;
    for (let k = 0; k < 4; k++) {
      if (DY[k]! * level.w + DX[k]! === d) dir = k;
    }
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += STEP[dir]!;
    player = from;
    boxes = nextBoxes;
  }
  return moves;
}

function chainLength(parent: number[], found: number): number {
  let n = 0;
  for (let s = found; s > 0; s = parent[s]!) n++;
  return n;
}

// Push-optimal breadth-first search. `exitDoor` is -1 for a plain search,
// or a square the player must be able to reach once the last box lands
// (used when a room's first solution would strand the player inside it).
// Returns null when the state cap or the run budget runs out, or when the
// level has no solution.
function solvePushBfs(
  level: Level,
  maxStates: number,
  budget: { used: number; cap: number },
  exitDoor: number,
): SearchResult | null {
  const kit = makeKit(level, budget);
  const { w, wall, goalSet } = level;
  const size = wall.length;
  const { live, occupied, seen, mark, stack } = kit;
  if (level.boxes.some((b) => !live[b])) return null;
  if (allOnGoals(level, level.boxes)) {
    if (exitDoor < 0) return { pushes: 0, moves: '' };
    for (const b of level.boxes) occupied[b] = 1;
    const ok = canReach(level, occupied, level.player, exitDoor, seen, ++kit.stamp, stack);
    for (const b of level.boxes) occupied[b] = 0;
    return ok ? { pushes: 0, moves: '' } : null;
  }

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const index = new Map<number | string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  index.set(
    stateKey(level.boxes, floodRegion(level, occupied, level.player, seen, ++kit.stamp, stack), size),
    0,
  );
  for (const b of level.boxes) occupied[b] = 0;

  let found = -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    floodRegion(level, occupied, statesPlayer[head]!, seen, ++kit.stamp, stack);
    const here = kit.stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenAt(level, occupied, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        let exitOk = true;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          key = stateKey(next, floodRegion(level, occupied, b, mark, ++kit.stamp, stack), size);
          if (exitDoor >= 0 && allOnGoals(level, next)) {
            exitOk = canReach(level, occupied, b, exitDoor, mark, ++kit.stamp, stack);
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || !exitOk || index.has(key)) continue;
        if (statesBoxes.length >= maxStates || budget.used >= budget.cap) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        budget.used++;
        index.set(key, statesBoxes.length);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushTo.push(t);
        if (allOnGoals(level, next!)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;
  const moves = rebuildMoves(kit, statesBoxes, parent, pushFrom, pushTo, found);
  return { pushes: chainLength(parent, found), moves };
}

// Push-based A* with the min-matching heuristic. Optimal when it runs to
// completion; used only as a fallback when BFS runs out of states.
function solvePushAstar(
  level: Level,
  maxStates: number,
  exitDoor: number,
): SearchResult | null {
  const kit = makeKit(level, { used: 0, cap: Number.MAX_SAFE_INTEGER });
  const { w, wall } = level;
  const size = wall.length;
  const { live, occupied, seen, mark, stack } = kit;
  if (level.boxes.some((b) => !live[b])) return null;
  if (allOnGoals(level, level.boxes)) {
    if (exitDoor < 0) return { pushes: 0, moves: '' };
    for (const b of level.boxes) occupied[b] = 1;
    const ok = canReach(level, occupied, level.player, exitDoor, seen, ++kit.stamp, stack);
    for (const b of level.boxes) occupied[b] = 0;
    return ok ? { pushes: 0, moves: '' } : null;
  }

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const gScore: number[] = [0];
  const best = new Map<number | string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  best.set(
    stateKey(level.boxes, floodRegion(level, occupied, level.player, seen, ++kit.stamp, stack), size),
    0,
  );
  for (const b of level.boxes) occupied[b] = 0;

  // Binary heap of [f, g, state]; smallest f first, deeper g first on ties.
  const heap: [number, number, number][] = [[matchingHeuristic(level, level.boxes), 0, 0]];
  const less = (a: [number, number, number], b2: [number, number, number]): boolean =>
    a[0] < b2[0] || (a[0] === b2[0] && (a[1] > b2[1] || (a[1] === b2[1] && a[2] < b2[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  let found = -1;
  while (heap.length > 0 && found < 0) {
    const [, g, s] = heapPop();
    if (gScore[s] !== g) continue;
    const boxes = statesBoxes[s]!;
    if (allOnGoals(level, boxes)) {
      if (exitDoor < 0) {
        found = s;
        break;
      }
      for (const b of boxes) occupied[b] = 1;
      const ok = canReach(level, occupied, statesPlayer[s]!, exitDoor, seen, ++kit.stamp, stack);
      for (const b of boxes) occupied[b] = 0;
      if (ok) {
        found = s;
        break;
      }
      continue;
    }
    for (const b of boxes) occupied[b] = 1;
    floodRegion(level, occupied, statesPlayer[s]!, seen, ++kit.stamp, stack);
    const here = kit.stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenAt(level, occupied, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          key = stateKey(next, floodRegion(level, occupied, b, mark, ++kit.stamp, stack), size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= g + 1) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        best.set(key, g + 1);
        const ns = statesBoxes.length;
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        gScore.push(g + 1);
        heapPush([g + 1 + matchingHeuristic(level, next!), g + 1, ns]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;
  const moves = rebuildMoves(kit, statesBoxes, parent, pushFrom, pushTo, found);
  return { pushes: chainLength(parent, found), moves };
}

interface Room {
  door: number; // room square next to the hallway: the way in and out
  entry: number; // hallway square next to the door: where a room solve starts
  boxes: number[];
  goals: number[];
}

// Split a level into rooms: cut the hallway row's dead squares (a box there
// can reach no goal) out of the floor; each remaining component holding
// boxes is a room. Null when the level does not split cleanly.
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const comps: number[][] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = comps.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = comps.length;
          squares.push(q);
        }
      }
    }
    comps.push(squares);
  }
  if (goals.some((g) => part[g]! < 0)) return null;
  const rooms: Room[] = [];
  for (let ci = 0; ci < comps.length; ci++) {
    const squares = comps[ci]!;
    const rBoxes = boxes.filter((b) => part[b] === ci);
    const rGoals = goals.filter((g) => part[g] === ci);
    if (rBoxes.length !== rGoals.length) return null;
    if (rBoxes.length === 0) continue;
    // The door is the room square next to a floor square of the hallway
    // row that belongs to no component of this room.
    let door = -1;
    let entry = -1;
    for (const s of squares) {
      for (let k = 0; k < 4; k++) {
        const n = s + DY[k]! * w + DX[k]!;
        if (Math.floor(n / w) === row && !wall[n] && part[n] !== ci) {
          door = s;
          entry = n;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) return null;
    rooms.push({ door, entry, boxes: rBoxes, goals: rGoals });
  }
  return rooms;
}

// Can the player walk out to the room's door once `solved` has landed every
// box on its goal? The player starts the room at `room.entry`.
function canExit(sub: Level, room: Room, solved: SearchResult): boolean {
  const end = endSquare(sub.w, room.entry, solved.moves);
  const occupied = new Uint8Array(sub.wall.length);
  for (const g of room.goals) occupied[g] = 1;
  const seen = new Int32Array(sub.wall.length);
  const stack = new Int32Array(sub.wall.length);
  return canReach(sub, occupied, end, room.door, seen, 1, stack);
}

// Solve one room with the other rooms' boxes and goals standing as walls.
// The player starts at the room's entry square in the hallway. Prefers a
// push-optimal solution the player can walk out of; falls back to a
// push-optimal-with-exit search, then to A*, and finally to the plain
// optimal solution even if the player is stranded (it may still serve if
// this room is solved last).
function solveRoom(
  level: Level,
  room: Room,
  rooms: Room[],
  budget: { used: number; cap: number },
): SearchResult | null {
  const wall = level.wall.slice();
  for (const other of rooms) {
    if (other === room) continue;
    for (const b of other.boxes) wall[b] = 1;
    for (const g of other.goals) wall[g] = 1;
  }
  const sub: Level = {
    w: level.w,
    h: level.h,
    wall,
    goals: room.goals.slice().sort((a, b) => a - b),
    goalSet: level.goalSet,
    boxes: room.boxes.slice().sort((a, b) => a - b),
    player: room.entry,
  };
  const plain = solvePushBfs(sub, ROOM_CAP, budget, -1);
  if (plain !== null && canExit(sub, room, plain)) return plain;
  const withExit = solvePushBfs(sub, ROOM_CAP, budget, room.door);
  if (withExit !== null) return withExit;
  const astar = solvePushAstar(sub, ASTAR_ROOM_CAP, room.door);
  if (astar !== null) return astar;
  return plain;
}

// Walk the player through the rooms in `order`, solving each as it comes,
// and stitch every room's move string into one answer.
function stitch(
  level: Level,
  rooms: Room[],
  solved: SearchResult[],
  order: number[],
): SearchResult {
  const occupied = new Uint8Array(level.wall.length);
  for (const b of level.boxes) occupied[b] = 1;
  let player = level.player;
  let moves = '';
  let pushes = 0;
  for (const ri of order) {
    const room = rooms[ri]!;
    const s = solved[ri]!;
    moves += walkTo(level, occupied, player, room.entry);
    player = room.entry;
    moves += s.moves;
    pushes += s.pushes;
    for (const b of room.boxes) occupied[b] = 0;
    for (const g of room.goals) occupied[g] = 1;
    player = endSquare(level.w, player, s.moves);
  }
  return { pushes, moves };
}

// Solve a level room by room. Tries the rooms in component order, then in
// reverse, so a walk that fails one way gets a second chance.
function solveByRooms(level: Level, budget: { used: number; cap: number }): SearchResult | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  const solved: SearchResult[] = [];
  for (const room of rooms) {
    const s = solveRoom(level, room, rooms, budget);
    if (s === null) return null;
    solved.push(s);
  }
  const orders = [rooms.map((_, i) => i), rooms.map((_, i) => i).reverse()];
  for (const order of orders) {
    try {
      return stitch(level, rooms, solved, order);
    } catch {
      // A walk failed; try the other order, then give up on rooms.
    }
  }
  return null;
}

function solveLevel(text: string): string {
  try {
    const level = parseLevel(text);
    if (level.boxes.length === 0) return '';
    const budget = { used: 0, cap: RUN_CAP };
    let solved = solveByRooms(level, budget);
    if (solved === null) {
      // Whole-level push BFS, then whole-level A*.
      solved =
        solvePushBfs(level, WHOLE_CAP, budget, -1) ?? solvePushAstar(level, WHOLE_CAP, -1);
    }
    return solved === null ? '' : solved.moves;
  } catch {
    // One bad instance must never sink the other four.
    return '';
  }
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
