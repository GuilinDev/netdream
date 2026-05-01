import { AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";

// Subtle animated gradient + drifting particles + a real Pexels particle
// network video looped at low opacity behind the foreground content.
//
// We keep the SVG particles too — they fill in the gaps and give the
// figure-side of the screen visual texture even where the video is dark.

export const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  const particles = Array.from({ length: 60 }, (_, i) => {
    const seed = (i * 9301 + 49297) % 233280;
    const x = (seed / 233280) * width;
    const y = ((seed * 17) % 233280 / 233280) * height;
    const phase = (i * 13) % 100;
    const drift = Math.sin((frame + phase) / 60) * 12;
    const opacity = 0.12 + 0.08 * Math.sin((frame + phase * 2) / 80);
    return { x: x + drift, y, opacity, r: 1.5 + (i % 3) };
  });

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(180deg, ${palette.bgTop} 0%, ${palette.bgBottom} 100%)`,
      }}
    >
      {/* Pexels particle-network video, very low opacity, sits behind everything */}
      <AbsoluteFill style={{ opacity: 0.32 }}>
        <OffthreadVideo
          src={staticFile("particles.mp4")}
          muted
          loop
          startFrom={0}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </AbsoluteFill>

      {/* SVG particles overlaid for extra liveliness */}
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

      {/* Soft white wash layer to keep foreground readable */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at center, rgba(255,255,255,0.65) 0%, rgba(255,255,255,0.25) 100%)`,
        }}
      />
    </AbsoluteFill>
  );
};
