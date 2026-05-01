import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";
import { timing, sceneFrame } from "../lib/timing";

// 12–22 s (frames 360-660) — Module C2: Imagination + Safety Filter
//
// Beats:
//   0-60   Many candidate sequences materialize (grid of colored cells)
//   60-150 One sequence highlighted, rolled out 5 steps with rising risk
//   150-220 Safety filter: ✗ ✗ ✗ ✓ ✗ — chosen ✓ glows
//   220-300 Caption: "Imagine futures, reject unsafe plans"

const ROLLOUT_COLORS = [
  palette.riskLow,
  palette.riskLow,
  palette.riskMid,
  palette.riskHigh,
  palette.riskHigh,
];

export const ImaginationPlanner: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = sceneFrame(frame, timing.c2Start);

  // 1. Candidate sequences materialize
  const candidatesIn = (row: number, col: number) =>
    spring({
      frame: local - 5 - row * 4 - col * 2,
      fps,
      config: { damping: 14, stiffness: 110 },
    });

  // 2. Pull one candidate out for rollout
  const rolloutAppear = (h: number) =>
    spring({
      frame: local - 80 - h * 12,
      fps,
      config: { damping: 12, stiffness: 100 },
    });

  // 3. Safety filter markers
  const markerAppear = (i: number) =>
    spring({
      frame: local - 165 - i * 8,
      fps,
      config: { damping: 13, stiffness: 110 },
    });
  const chosenIdx = 3; // the ✓ that gets selected
  const chosenGlow =
    1.0 + 0.15 * Math.sin((local - 200) / 6) * (local > 200 ? 1 : 0);

  // Caption
  const captionOpacity = interpolate(local, [225, 250, 290, 300], [0, 1, 1, 0], {
    extrapolateRight: "clamp",
  });

  const sceneOut = interpolate(local, [285, 300], [1, 0], {
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ position: "absolute", inset: 0, opacity: sceneOut }}>
      <SectionHeader text="(C2) Imagination-Based Planner" subFrame={local} />

      <svg width="100%" height="100%" viewBox="0 0 1920 1080">
        <defs>
          <marker
            id="arrow2"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill={palette.borderGray} />
          </marker>
        </defs>

        {/* === LEFT: Candidate Scaling Sequences (8 rows × 5 cells) === */}
        <text
          x={300}
          y={250}
          fontFamily="Inter, sans-serif"
          fontSize={22}
          fontWeight={700}
          fill={palette.textPrimary}
          textAnchor="middle"
        >
          Candidate Scaling Sequences
        </text>
        {[...Array(8)].map((_, row) =>
          [...Array(5)].map((__, col) => {
            const a = candidatesIn(row, col);
            const colorIdx = (row + col) % 3;
            const fill =
              colorIdx === 0
                ? palette.paleMint
                : colorIdx === 1
                ? palette.paleCoral
                : palette.palePink;
            const x = 200 + col * 36;
            const y = 280 + row * 26;
            return (
              <rect
                key={`${row}-${col}`}
                x={x}
                y={y}
                width={32 * a}
                height={22 * a}
                rx={4}
                fill={fill}
                stroke={palette.borderLight}
                strokeWidth={0.8}
                opacity={a}
              />
            );
          })
        )}
        <text
          x={300}
          y={520}
          fontFamily="Inter, sans-serif"
          fontSize={18}
          fontStyle="italic"
          fill={palette.textMuted}
          textAnchor="middle"
          opacity={interpolate(local, [40, 60], [0, 1], {
            extrapolateRight: "clamp",
          })}
        >
          scale up / no-op / scale down
        </text>

        {/* === ARROW from candidates → rollout === */}
        <line
          x1={420}
          y1={400}
          x2={620}
          y2={400}
          stroke={palette.borderGray}
          strokeWidth={2}
          markerEnd="url(#arrow2)"
          opacity={interpolate(local, [70, 90], [0, 1], {
            extrapolateRight: "clamp",
          })}
        />

        {/* === CENTER: Rollout via World Model === */}
        <text
          x={1000}
          y={250}
          fontFamily="Inter, sans-serif"
          fontSize={22}
          fontWeight={700}
          fill={palette.textPrimary}
          textAnchor="middle"
          opacity={interpolate(local, [70, 95], [0, 1], {
            extrapolateRight: "clamp",
          })}
        >
          Rollout via World Model
        </text>
        {[0, 1, 2, 3, 4].map((h) => {
          const a = rolloutAppear(h);
          const x = 700 + h * 130;
          return (
            <g key={h}>
              {/* ĉ indicator (small circle, color along risk gradient) */}
              <circle
                cx={x + 40}
                cy={310}
                r={14 * a}
                fill={ROLLOUT_COLORS[h]}
                stroke={palette.borderGray}
                strokeWidth={1}
                opacity={a}
              />
              <text
                x={x + 40}
                y={315}
                textAnchor="middle"
                fontSize={12}
                fontStyle="italic"
                fill={palette.textPrimary}
                opacity={a}
              >
                ĉ
              </text>
              {/* Snapshot square */}
              <rect
                x={x}
                y={350}
                width={80 * a}
                height={80 * a}
                rx={8}
                fill={ROLLOUT_COLORS[h]}
                stroke={palette.borderGray}
                strokeWidth={1.2}
                opacity={a * 0.7}
              />
              {/* Step label */}
              <text
                x={x + 40}
                y={460}
                textAnchor="middle"
                fontSize={14}
                fontStyle="italic"
                fill={palette.textSecondary}
                opacity={a}
              >
                X̂ₜ₊{h}
              </text>
              {h < 4 ? (
                <line
                  x1={x + 90}
                  y1={390}
                  x2={x + 130 - 8}
                  y2={390}
                  stroke={palette.borderGray}
                  strokeWidth={2}
                  markerEnd="url(#arrow2)"
                  opacity={a}
                />
              ) : null}
            </g>
          );
        })}

        {/* === ARROW rollout → safety filter === */}
        <line
          x1={1340}
          y1={400}
          x2={1500}
          y2={400}
          stroke={palette.borderGray}
          strokeWidth={2}
          markerEnd="url(#arrow2)"
          opacity={interpolate(local, [150, 170], [0, 1], {
            extrapolateRight: "clamp",
          })}
        />

        {/* === RIGHT: Safety Filter ===  */}
        <text
          x={1660}
          y={250}
          fontFamily="Inter, sans-serif"
          fontSize={22}
          fontWeight={700}
          fill={palette.textPrimary}
          textAnchor="middle"
          opacity={interpolate(local, [150, 175], [0, 1], {
            extrapolateRight: "clamp",
          })}
        >
          Safety Filter
        </text>
        {[
          { mark: "✗", color: "#C0392B", chosen: false },
          { mark: "✗", color: "#C0392B", chosen: false },
          { mark: "✗", color: "#C0392B", chosen: false },
          { mark: "✓", color: palette.accentBlue, chosen: true },
          { mark: "✗", color: "#C0392B", chosen: false },
          { mark: "✓", color: "#27AE60", chosen: false },
        ].map((m, i) => {
          const a = markerAppear(i);
          const y = 290 + i * 50;
          const isChosen = m.chosen;
          const scale = isChosen ? chosenGlow : 1;
          return (
            <g key={i} opacity={a} transform={`translate(1660, ${y}) scale(${scale})`}>
              {isChosen ? (
                <rect
                  x={-30}
                  y={-30}
                  width={60}
                  height={60}
                  rx={8}
                  fill="none"
                  stroke={palette.accentBlue}
                  strokeWidth={3}
                />
              ) : null}
              <text
                x={0}
                y={10}
                textAnchor="middle"
                fontSize={36}
                fontWeight={700}
                fill={m.color}
              >
                {m.mark}
              </text>
            </g>
          );
        })}
        <text
          x={1660}
          y={620}
          textAnchor="middle"
          fontFamily="Inter, sans-serif"
          fontSize={20}
          fontWeight={700}
          fill={palette.accentBlue}
          opacity={interpolate(local, [195, 220], [0, 1], {
            extrapolateRight: "clamp",
          })}
        >
          best safe plan
        </text>
        <text
          x={1660}
          y={650}
          textAnchor="middle"
          fontFamily="Inter, sans-serif"
          fontSize={16}
          fontStyle="italic"
          fill={palette.textSecondary}
          opacity={interpolate(local, [205, 225], [0, 1], {
            extrapolateRight: "clamp",
          })}
        >
          → execute first action
        </text>
      </svg>

      <div
        style={{
          position: "absolute",
          bottom: 100,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: captionOpacity,
          fontSize: 32,
          color: palette.textSecondary,
          fontFamily: "Inter, sans-serif",
          fontStyle: "italic",
        }}
      >
        Imagine many possible futures — reject the unsafe ones.
      </div>
    </div>
  );
};

const SectionHeader: React.FC<{ text: string; subFrame: number }> = ({
  text,
  subFrame,
}) => {
  const opacity = interpolate(subFrame, [0, 15, 285, 300], [0, 1, 1, 0], {
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        position: "absolute",
        top: 60,
        left: 100,
        opacity,
        fontSize: 44,
        fontWeight: 800,
        color: palette.accentBlueDeep,
        fontFamily: "Inter, sans-serif",
        letterSpacing: -1,
      }}
    >
      {text}
    </div>
  );
};
