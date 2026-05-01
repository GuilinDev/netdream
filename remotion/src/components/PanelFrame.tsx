import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

// Scene-wide panel background — saturated enough to be visible against
// the busy particle background.

type Variant = "blue" | "lavender" | "mint";

const FILLS: Record<Variant, string> = {
  blue: "#DCE9F4",      // slightly more saturated than panelBlueBg
  lavender: "#E8DDF0",  // slightly more saturated than panelLavBg
  mint: "#DFEEDE",      // slightly more saturated than panelMintBg
};

const BORDERS: Record<Variant, string> = {
  blue: "#7AA8D6",
  lavender: "#A688B8",
  mint: "#7CB87E",
};

export const PanelFrame: React.FC<{
  variant: Variant;
  fadeStart?: number;
  fadeEnd?: number;
  fadeOutStart?: number;
  fadeOutEnd?: number;
  localFrame?: number;
}> = ({
  variant,
  fadeStart = 0,
  fadeEnd = 15,
  fadeOutStart = 9999,
  fadeOutEnd = 99999,
  localFrame,
}) => {
  const globalFrame = useCurrentFrame();
  const f = localFrame ?? globalFrame;
  const opacity = interpolate(
    f,
    [fadeStart, fadeEnd, fadeOutStart, fadeOutEnd],
    [0, 1, 1, 0],
    { extrapolateRight: "clamp", extrapolateLeft: "clamp" }
  );
  return (
    <AbsoluteFill
      style={{
        opacity,
        padding: 60,
        pointerEvents: "none",
        zIndex: 0,
      }}
    >
      <div
        style={{
          width: "100%",
          height: "100%",
          background: FILLS[variant],
          border: `2px solid ${BORDERS[variant]}`,
          borderRadius: 24,
          boxShadow: "0 8px 40px rgba(0,0,0,0.10)",
        }}
      />
    </AbsoluteFill>
  );
};
