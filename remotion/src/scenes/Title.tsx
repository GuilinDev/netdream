import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";
import { timing, sceneFrame } from "../lib/timing";

// 0–3 s — Title and hook
//
// Layout:
//   [hook line top]   "Reactive autoscaling is always one step behind."
//   [big title]       "NetDream"
//   [subtitle]        "Graph World Models for Safe Kubernetes Autoscaling"

export const Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = sceneFrame(frame, timing.titleStart);

  const hookOpacity = interpolate(local, [0, 12, 75, 90], [0, 1, 1, 0], {
    extrapolateRight: "clamp",
  });
  const hookY = interpolate(local, [0, 20], [-20, 0], {
    extrapolateRight: "clamp",
  });

  const titleSpring = spring({
    frame: local - 18,
    fps,
    config: { damping: 14, stiffness: 90, mass: 1.2 },
  });
  const titleScale = interpolate(titleSpring, [0, 1], [0.85, 1]);
  const titleOpacity = interpolate(local, [18, 32, 80, 90], [0, 1, 1, 0], {
    extrapolateRight: "clamp",
  });

  const subSpring = spring({
    frame: local - 36,
    fps,
    config: { damping: 14, stiffness: 90 },
  });
  const subOpacity = interpolate(local, [36, 50, 80, 90], [0, 1, 1, 0], {
    extrapolateRight: "clamp",
  });
  const subY = interpolate(subSpring, [0, 1], [10, 0]);

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 40,
        textAlign: "center",
      }}
    >
      <div
        style={{
          opacity: hookOpacity,
          transform: `translateY(${hookY}px)`,
          fontSize: 28,
          color: palette.textMuted,
          fontStyle: "italic",
          letterSpacing: 1,
          fontFamily: "Inter, Helvetica, Arial, sans-serif",
        }}
      >
        Reactive autoscaling is always one step behind.
      </div>

      <div
        style={{
          opacity: titleOpacity,
          transform: `scale(${titleScale})`,
          fontSize: 200,
          fontWeight: 800,
          letterSpacing: -4,
          color: palette.accentBlueDeep,
          fontFamily: "Inter, Helvetica, Arial, sans-serif",
          lineHeight: 1,
        }}
      >
        NetDream
      </div>

      <div
        style={{
          opacity: subOpacity,
          transform: `translateY(${subY}px)`,
          fontSize: 36,
          color: palette.textSecondary,
          fontFamily: "Inter, Helvetica, Arial, sans-serif",
          fontWeight: 400,
          maxWidth: 1200,
        }}
      >
        Graph World Models for Safe Kubernetes Autoscaling
      </div>
    </div>
  );
};
