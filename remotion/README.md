# NetDream Animation (Remotion)

A 30-second explainer animation for **NetDream** — the graph world model
for safe Kubernetes autoscaling described in the main paper.

## Storyboard

| Time | Frames | Scene | Content |
|------|--------|-------|---------|
| 0–3 s   | 0–90    | Title           | "NetDream" + tagline + hook line |
| 3–12 s  | 90–360  | (C1) Graph World Model     | 11-node service graph spring-in, encoder + GAT pulse, three prediction heads emerge |
| 12–22 s | 360–660 | (C2) Imagination Planner   | K candidate sequences materialize, one rolled out 5 steps with rising risk, safety filter ✗✗✗✓✗ — chosen ✓ glows accent-blue |
| 22–27 s | 660–810 | (C3) Online Deployment     | Kubernetes Controller → Cluster (pods scale up) → Metrics, dashed feedback loop |
| 27–30 s | 810–900 | Outro                      | Cost–violations Pareto plot animates in, NetDream-Safe pulses, 2.7× headline |

## How to run

```bash
cd remotion
npm install              # one-time
npm start                # opens Remotion Studio at http://localhost:3000
                          # — live preview, scrub timeline
```

## Render to MP4

```bash
npm run build            # writes out/netdream.mp4 (1920x1080, 30 fps, h264)
```

## Render to GIF (smaller, for README embed)

```bash
npm run build-gif        # writes out/netdream.gif
```

## Stock footage and audio (already integrated)

`remotion/public/` ships with four Pexels-licensed assets that the
animation already uses:

| File | Origin | Where it appears |
|------|--------|------------------|
| `particles.mp4`    | abstract particle network | background through every scene (opacity 0.32) |
| `icosahedron.mp4`  | wireframe 3-D geometry    | hero overlay during the Title (0–3 s) and the Outro (27–30 s) |
| `terminal.mp4`     | scrolling code/log lines  | brief overlay around the "Logged Transitions" cylinder in (C1) |
| `bgm.mp3`          | ambient electronic music  | full-track background, volume 0.18 |

To swap or remove, edit `src/components/Background.tsx`,
`src/scenes/Title.tsx`, `src/scenes/GraphWorldModel.tsx`,
`src/scenes/Outro.tsx`, and the `<Audio>` tag in `src/NetDream.tsx`.

## Color palette

Aligned with the paper's Figure 2 (`paper/figures/fig_overview.pdf`):
pale pastel watercolor + accent blue `#5B8DBE`. See `src/lib/palette.ts`.

## Customising timing

All scene boundaries live in `src/lib/timing.ts`. Tweak there and the
top-level composition will re-render automatically in Remotion Studio.
