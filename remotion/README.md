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

## Optional: drop in stock footage

The animation is fully self-contained — no external assets required.
But if you want extra texture, the codebase is structured to consume
optional Pexels-style clips:

1. Download free clips from https://www.pexels.com/videos:
   - `data center` flythrough → for the (C3) deployment scene
   - `abstract data network` particles → background overlay
   - `code on screen` → 1-second flash during (C1)
2. Drop them into `remotion/public/` (named e.g. `datacenter.mp4`,
   `particles.mp4`, `code.mp4`).
3. Edit `src/components/Background.tsx` and the relevant scene to import
   `<Video src={staticFile("datacenter.mp4")} />` from Remotion.

For background music, Pexels also has free royalty-free audio. Place a
30-second clip at `public/bgm.mp3` and add `<Audio src={staticFile("bgm.mp3")} />`
in `NetDream.tsx`.

## Color palette

Aligned with the paper's Figure 2 (`paper/figures/fig_overview.pdf`):
pale pastel watercolor + accent blue `#5B8DBE`. See `src/lib/palette.ts`.

## Customising timing

All scene boundaries live in `src/lib/timing.ts`. Tweak there and the
top-level composition will re-render automatically in Remotion Studio.
