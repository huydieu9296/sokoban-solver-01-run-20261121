# Sokoban solver — push-optimal, room by room

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/huydieu9296/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/huydieu9296/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

Entry for the make.fun bounty [Build the best Sokoban solver](https://make.fun/bounty/sokoban-solver-01-run-20261121) — $100 over 10 places.

## How it works

Every level is a hallway with one to four rooms hanging off it, and a box
that leaves its room can never come back, so the rooms are independent
puzzles whose push-optima add up to the level's P*. The solver:

1. Splits the level into rooms by cutting the hallway row's dead squares
   out of the floor.
2. Solves each room on its own with a breadth-first search over pushes —
   a state is the sorted box squares plus the player's reachable region —
   which is push-optimal by construction.
3. Stitches the rooms' move strings together, walking the player between
   rooms through the hallway.

Pruning is all safe (dead squares, frozen 2x2 blocks, a transposition
table), and the search is bounded by fixed state counts, never by the
clock, so two runs print identical bytes. Fallbacks, in order: a
push-optimal search that also requires the player to be able to walk out
of the room, push A* with an admissible min-matching heuristic, then a
whole-level search.

## Practice score

1,000,000 / 1,000,000 on the kit's practice set (all five instances at
1,000/1,000 pushes-optimal), verified with the kit's local grader
(`node cli/makefun.mjs grade-local --kit . --bot <repo> --driver process`).
The house baseline (plain-astar 1.0.0) scores 200,000 on the same set.

## Files

- `main.ts` — the solver (Node 22, erasable TypeScript only, no imports
  beyond Node builtins, no network).
- `LICENSE` — MIT.
