import { AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";

// Solid pale background + very subtle particle accent. Kept very low so
// the busy particle video does not wash out the foreground SVG content
// that each scene renders on top.

export const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  const particles = Array.from({ length: 40 }, (_, i) => {
    const seed = (i * 9301 + 49297) % 233280;
    const x = (seed / 233280) * width;
    const y = ((seed * 17) % 233280 / 233280) * height;
    const phase = (i * 13) % 100;
    const drift = Math.sin((frame + phase) / 60) * 12;
    const opacity = 0.10 + 0.06 * Math.sin((frame + phase * 2) / 80);
    return { x: x + drift, y, opacity, r: 1.5 + (i % 3) };
  });

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(180deg, #FAFCFF 0%, #EFF4F9 100%)`,
      }}
    >
      {/* Very subtle Pexels particle video */}
      <AbsoluteFill style={{ opacity: 0.08 }}>
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

      {/* SVG accent particles */}
      <svg
        width={width}
        height={height}
        style={{ position: "absolute", inset: 0 }}
      >
        {particles.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={p.r}
            fill={palette.accentBlue}
            opacity={p.opacity}
          />
        ))}
      </svg>
    </AbsoluteFill>
  );
};
