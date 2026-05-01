// Master timeline — 900 frames @ 30 fps = 30 seconds.
// Each scene is wrapped in a <Sequence from={...}>, so inside the scene
// component `useCurrentFrame()` ALREADY returns a sequence-local frame
// starting at 0. Each scene's local timeline is therefore:
//
//   Title: 0..89 (3 s)
//   C1:    0..269 (9 s)
//   C2:    0..299 (10 s)
//   C3:    0..149 (5 s)
//   Outro: 0..89  (3 s)
//
// `sceneFrame` is kept as an identity helper for backwards compatibility
// with existing scene code; it just returns the input frame unchanged.

export const timing = {
  // Title / hook
  titleStart: 0,
  titleEnd: 90,

  // C1: Graph World Model
  c1Start: 90,
  c1End: 360,

  // C2: Imagination + Safety Filter
  c2Start: 360,
  c2End: 660,

  // C3: Online Deployment
  c3Start: 660,
  c3End: 810,

  // Outro / result
  outroStart: 810,
  outroEnd: 900,
} as const;

// Scene-local frame: inside a <Sequence>, useCurrentFrame() already
// returns the frame relative to the sequence start. So this helper
// is now an identity — kept only so existing call sites don't break.
export const sceneFrame = (frame: number, _sceneStart: number): number =>
  Math.max(0, frame);
