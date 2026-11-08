# Ocarina Dynamic FPS add-on

`Dynamic FPS by Dark (built-in)` is a game Add-On, enabled by default for Ocarina.
It targets title `0400486024DEC000`, module `main`, build ID
`E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348` only. Max Resolution remains a
separate, default-on Add-On. Either can be disabled independently in Add-Ons;
the existing disabled-add-on configuration persists that choice.

## Credit and source

Dark, an Eglantine developer, wrote the routine. This port starts from the call and
120-byte routine from Frizti's `oot-gather-fix` commit
`765659af4ccb7201d1145c57f3b45ce84653cea0`. That commit explicitly states that
Dark gave permission to bundle it. Frizti supplied the built-in integration.

Dark's routine has SHA-256
`3c15dfb57f748328ad7d1c1675941dc2ace32a5e7b10907368ceac6a9a8c5ee1`.
This port's routine changes it (see Timing limits) and has SHA-256
`9ed3480a8faa37f59c75087bc48a4bcdf0aab68252d52b128f98e26bccbeff1a`.
The call's SHA-256 is
`e4d3f65ad1b568b557f02b9a087a9c304240ac70d6ef474dac83a1f97737515e`.
This port adds original-byte and memory-layout guards, and
transactional validation before writing either record.

## Verified original layout

An independently extracted original `main.text` has SHA-256
`f71eb7b75b11dcc8d955f3f6e987a098573288d9ce41aa22a96272541e858c5f`, matching
the text hash in its NSO header. Its instruction at module offset `0x6E6CC4`
is `28 18 20 1E`, or `fdiv s8, s1, s0`. The patch replaces it with
`bl 0x3C46200` and executes the original division first in the new routine.
The enclosing function saves and restores its return address. The continuation
does not consume the routine's clobbered scratch registers or flags before
intervening calls or redefinition.

The original text ends at `0x3C461D0`; read-only data starts at `0x3C47000`.
The routine occupies `0x3C46200` through `0x3C4629F`, in executable alignment
padding, not original text. `AppLoader_NSO::LoadModule` zero-initializes the
module image before copying segments and rounds the text mapping to a page.
The loader checks that this space is still zero-filled executable padding
before installing the routine.

Data ends at `0x4D46B80` and BSS ends at `0x4EA0000`. The routine stores its
previous timestamp and smoothed delta at `0x4E9FE00` and `0x4E9FE08`, and the
nominal and final frame step, as floats, at `0x4E9FE10` and `0x4E9FE14` so that
a debugger or `EDEN_NX2_PEEK` can read them. The
loader requires both words to fit BSS and remain zero-initialized. These
checks establish addressability and initialization, not exclusive ownership
of that storage throughout gameplay.

Any changed user-mod bytes in either patch record or the timer state take
precedence over the entire built-in patch. An incompatible build, original
instruction, layout, or truncated module leaves both patch records untouched.

## Timing limits

The routine reads `CNTPCT_EL0` and `CNTFRQ_EL0`. After its first call, it caps
the timestamp difference at `CNTFRQ / 20`, or 50 ms, and then at 1.5 times the
smoothed value, then smooths it by adding one eighth of the signed difference
from the previous smoothed value.
It calculates `s2 = (smoothed_ticks / CNTFRQ) * s1`. It replaces the original
`s8 = s1 / s0` when `s2 > 1.03 * s8` or `s2 < 0.97 * s8`.
The original code selects `s1` as 30 or 60 and converts the integer field at
`x21 + 0x2C8` into `s0`, so the result is a frame-step scale, not seconds.

The 1.5 times clamp is this port's addition. Without it, a run of long frames
(a stutter, an area load) pushed the smoothed value to the 50 ms cap, and the
next good frames still used a step of about 3, so the game ran at up to three
times speed for a fraction of a second after each stall. With it a stall costs
a little game time instead. A lasting drop in frame rate still catches up: the
value can rise 6% a frame, doubling in about 12 frames.

Dark's routine kept the original step as a floor, so frames faster than the
nominal rate ran the game fast. This port scales the step down as well.
Measured with `EDEN_NX2_PEEK=4EBFE10`: the nominal step is 1.0, so `s1` and `s0`
are both 60 in play, and the final step tracks real time: 2.0 at 30 fps, 1.36 at
44 fps, 0.6 at 100 fps. A fixed walk ended at about the same place at 30 and
45 fps. Nothing has yet measured game speed in play above 60 fps, and any game
logic that counts frames instead of using this step would run fast there.

In Lichii, `CoreTiming::GetClockTicks` scales the counter for CPU Boost and
Overclock, while Dynarmic's advertised counter frequency stays unchanged.
Boost uses a 1.7 multiplier and Overclock uses 2. Enabling core-speed
synchronization can further divide the counter by the configured speed limit.
Those settings alter the elapsed seconds this routine calculates. CPU Clock
Normal and an unscaled counter are prerequisites for evaluating its timing;
the Add-On does not change global clock behavior or automatically override
the user's settings. Correct gameplay speed with the Add-On is unverified.

## Tests

The `[patch_manager]` tests call production `PatchManager::PatchNSO` with
synthetic module images and real IPS/IPS32 user mods. They cover default-on and
disable behavior, title/module/build gates, original-byte and layout
guards, truncated inputs, all-record user-mod precedence, and Max Resolution.
The `[addon_config]` tests round-trip the production configuration reader and
writer, including legacy configurations and repeated enable/disable cycles.

Run them from a clang/Ninja test build with:

```sh
cmake --build build --target tests --parallel 4
build/bin/tests '[patch_manager],[addon_config]'
```
