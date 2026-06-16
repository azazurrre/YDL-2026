# Aidahar

A small browser platformer in a single HTML file — no build step, no dependencies.
Named for the *Aidahar*, the dragon you fight through ten levels.

## Play

Open `mario.html` in any modern browser (double-click it, or run `open mario.html` on macOS).
Keep `player.png`, `monster.png`, and `castle.png` next to the HTML — they're the character, dragon, and yurt sprites.

## Controls

- **← / →** — move (the character faces the way it moves)
- **↑ / Space** — jump (press again in the air to **double jump**)
- **Enter** — advance menus / start a level
- **R** — restart the game
- **Jump to level** — the buttons under the game frame switch to any level

## Features

- **10 levels**, easiest to hardest, each framed by a start and end **castle** (a Kazakh yurt).
- **Level select** under the frame — jump straight to any of the ten levels.
- **Mystery `?` boxes** — bonk from below for coins, a size increase, a speed boost, or a dragon. Floating signs show what you won.
- **Two dragon types** — stomp normal ones once; **glowing armored ones** take **two** stomps.
- **Hazards** — deadly **spikes** and spinning **saw blades** (horizontal and vertical), plus bottomless pits.
- Image sprites for the player, dragon, and castle; lives, score, and little WebAudio sound effects for jumps, coins, stomps, and entering/leaving castles.
