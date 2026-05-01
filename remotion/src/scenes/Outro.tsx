import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";
import { timing, sceneFrame } from "../lib/timing";

// 27–30 s (frames 810-900) — Outro: Pareto frontier + headline result

const PARETO_POINTS = [
  { name: "HPA",            cost: 6.8,  viol: 27.6, color: "#A0A0A0", chosen: false },
  { name: "Random",         cost: 29.3, viol: 25.5, color: "#BBBBBB", chosen: false },
  { name: "PPO-500K",       cost: 35.6, viol: 29.7, color: "#B5D8B0", chosen: false },
  { name: "ND-Unsafe",      cost: 13.7, viol: 27.8, color: "#C9B4D8", chosen: false },
  { name: "HPA-min5",       cost: 42.9, viol: 25.0, color: "#F5C6C6", chosen: false },
  { name: "HPA-min3",       cost: 52.0, viol: 19.6, color: "#FFD9A8", chosen: false },
  { name: "NetDream-Safe",  cost: 24.0, viol: 19.4, color: palette.accentBlue, chosen: true },
];

const X_MIN = 0,
  X_MAX = 60;
const Y_MIN = 18,
  Y_MAX = 32;

// Map data → SVG coords
const PLOT = { left: 600, top: 250, width: 720, height: 540 };
const x2px = (c: number) =>
  PLOT.left + ((c - X_MIN) / (X_MAX - X_MIN)) * PLOT.width;
const y2px = (v: number) =>
  PLOT.top + (1 - (v - Y_MIN) / (Y_MAX - Y_MIN)) * PLOT.height;

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = sceneFrame(frame, timing.outroStart);

  // Plot axes appear
  const axesAppear = interpolate(local, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Each point appears in sequence
  const pointAppear = (i: number) =>
    spring({
      frame: local - 10 - i * 4,
      fps,
      config: { damping: 13, stiffness: 110 },
    });

  // NetDream-Safe glow pulse
  const ndPulse =
    1.0 + 0.15 * Math.sin((local - 50) / 6) * (local > 50 ? 1 : 0);

  // Result statistic
  const statOpacity = interpolate(local, [50, 70], [0, 1], {
    extrapolateRight: "clamp",
  });
  const statScale = spring({
    frame: local - 50,
    fps,
    config: { damping: 12, stiffness: 90 },
  });

  // URL fade-in
  const urlOpacity = interpolate(local, [70, 85], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <svg width="100%" height="100%" viewBox="0 0 1920 1080">
        {/* === Axes === */}
        <g opacity={axesAppear}>
          <line
            x1={PLOT.left}
            y1={PLOT.top + PLOT.height}
            x2={PLOT.left + PLOT.width}
            y2={PLOT.top + PLOT.height}
            stroke={palette.textPrimary}
            strokeWidth={2}
          />
          <line
            x1={PLOT.left}
            y1={PLOT.top}
            x2={PLOT.left}
            y2={PLOT.top + PLOT.height}
            stroke={palette.textPrimary}
            strokeWidth={2}
          />
          <text
            x={PLOT.left + PLOT.width / 2}
            y={PLOT.top + PLOT.height + 60}
            textAnchor="middle"
            fontSize={24}
            fontFamily="Inter, sans-serif"
            fill={palette.textPrimary}
          >
            Cost (replica-steps)
          </text>
          <text
            x={PLOT.left - 80}
            y={PLOT.top + PLOT.height / 2}
            textAnchor="middle"
            fontSize={24}
            fontFamily="Inter, sans-serif"
            fill={palette.textPrimary}
            transform={`rotate(-90 ${PLOT.left - 80} ${PLOT.top + PLOT.height / 2})`}
          >
            SLO Violations / episode
          </text>
        </g>

        {/* === Points === */}
        {PARETO_POINTS.map((p, i) => {
          const a = pointAppear(i);
          const radius = (p.chosen ? 22 : 14) * a * (p.chosen ? ndPulse : 1);
          return (
            <g key={p.name} opacity={a}>
              <circle
                cx={x2px(p.cost)}
                cy={y2px(p.viol)}
                r={radius}
                fill={p.color}
                stroke={palette.textPrimary}
                strokeWidth={p.chosen ? 3 : 1}
              />
              <text
                x={x2px(p.cost) + (p.chosen ? 28 : 18)}
                y={y2px(p.viol) + 6}
                fontSize={p.chosen ? 22 : 16}
                fontWeight={p.chosen ? 700 : 400}
                fontFamily="Inter, sans-serif"
                fill={p.chosen ? palette.accentBlueDeep : palette.textSecondary}
              >
                {p.name}
              </text>
            </g>
          );
        })}

        {/* "Pareto frontier" label */}
        <text
          x={x2px(15)}
          y={y2px(23)}
          fontSize={18}
          fontStyle="italic"
          fontFamily="Inter, sans-serif"
          fill={palette.accentBlue}
          opacity={interpolate(local, [40, 60], [0, 1], {
            extrapolateRight: "clamp",
          })}
        >
          ← Pareto frontier
        </text>
      </svg>

      {/* === Result statistic === */}
      <div
        style={{
          position: "absolute",
          top: 100,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: statOpacity,
          transform: `scale(${0.8 + 0.2 * statScale})`,
          fontFamily: "Inter, sans-serif",
        }}
      >
        <div
          style={{
            fontSize: 80,
            fontWeight: 800,
            color: palette.accentBlueDeep,
            letterSpacing: -2,
            lineHeight: 1,
          }}
        >
          2.7×
        </div>
        <div
          style={{
            fontSize: 28,
            color: palette.textSecondary,
            marginTop: 8,
          }}
        >
          more cost-efficient than the best static over-provisioning baseline
        </div>
      </div>

      {/* === Repo URL === */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: urlOpacity,
          fontFamily: "Inter, monospace",
          fontSize: 26,
          color: palette.textMuted,
        }}
      >
        github.com/anonymous/netdream
      </div>
    </div>
  );
};
