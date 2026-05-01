import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";

// Subtle animated gradient + drifting particle dots — provides the "tech"
// background feel without overpowering the foreground content.

export const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  // Particles: deterministic positions derived from a seed so the layout
  // looks busy but not chaotic.
  const particles = Array.from({ length: 80 }, (_, i) => {
    const seed = (i * 9301 + 49297) % 233280;
    const x = (seed / 233280) * width;
    const y = ((seed * 17) % 233280 / 233280) * height;
    const phase = (i * 13) % 100;
    const drift = Math.sin((frame + phase) / 60) * 12;
    const opacity = 0.18 + 0.12 * Math.sin((frame + phase * 2) / 80);
    return { x: x + drift, y, opacity, r: 1.5 + (i % 3) };
  });

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(180deg, ${palette.bgTop} 0%, ${palette.bgBottom} 100%)`,
      }}
    >
      <svg
        width={width}
        height={height}
        style={{ position: "absolute", inset: 0 }}
      >
        <defs>
          <radialGradient id="particleGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={palette.accentBlue} stopOpacity="1" />
            <stop offset="100%" stopColor={palette.accentBlue} stopOpacity="0" />
          </radialGradient>
        </defs>
        {particles.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={p.r}
            fill="url(#particleGlow)"
            opacity={p.opacity}
          />
        ))}
      </svg>
    </AbsoluteFill>
  );
};
