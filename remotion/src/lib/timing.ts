// Master timeline — 900 frames @ 30 fps = 30 seconds.
// Each scene returns its absolute start frame, end frame, and duration.

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

// Helper: relative frame within a scene window
export const sceneFrame = (
  globalFrame: number,
  sceneStart: number
): number => Math.max(0, globalFrame - sceneStart);
