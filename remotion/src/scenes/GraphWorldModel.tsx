import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";
import { timing, sceneFrame } from "../lib/timing";

// 3–12 s (frames 90-360) — Module C1: Graph World Model
//
// Beats:
//   90-150   Service graph forms (11 nodes spring in)
//   150-210  Logged transitions cylinder appears, arrow → encoder
//   210-300  Encoder + GAT pulse, then 3 heads emerge
//   300-360  Sub-caption: "Learn how scaling propagates"

const SERVICES = [
  { name: "api-gateway",          tier: 0, idx: 0 },
  { name: "request-router",       tier: 1, idx: 0 },
  { name: "inference-orchestrator", tier: 1, idx: 1 },
  { name: "recommender-model",    tier: 1, idx: 2 },
  { name: "ranker-model",         tier: 1, idx: 3 },
  { name: "ab-test-router",       tier: 1, idx: 4 },
  { name: "feature-store",        tier: 2, idx: 0 },
  { name: "embedding-service",    tier: 2, idx: 1 },
  { name: "result-aggregator",    tier: 2, idx: 2 },
  { name: "logging-service",      tier: 2, idx: 3 },
  { name: "vector-cache",         tier: 2, idx: 4 },
] as const;

const COLORS = [
  palette.paleBlue,    palette.paleLavender, palette.paleMint,
  palette.palePeach,   palette.palePink,     palette.paleYellow,
  palette.paleCoral,   palette.paleBlue,     palette.paleLavender,
  palette.paleMint,    palette.palePeach,
];

const EDGES: Array<[number, number]> = [
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [2, 7], [2, 8], [2, 9], [1, 6], [1, 10], [3, 6],
];

const nodePos = (tier: number, idx: number) => {
  const cx = 380; // center x
  const cyTier = [120, 240, 360];
  const xSpacing = 90;
  const tierLengths = [1, 5, 5];
  const len = tierLengths[tier];
  const x = cx + (idx - (len - 1) / 2) * xSpacing;
  return { x, y: cyTier[tier] };
};

export const GraphWorldModel: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = sceneFrame(frame, timing.c1Start);

  // Sub-scene timings (relative to C1 start = 0)
  const graphAppear = (i: number) =>
    spring({
      frame: local - 5 - i * 4,
      fps,
      config: { damping: 12, stiffness: 100 },
    });

  const edgeAppear = (i: number) =>
    interpolate(local, [60 + i * 3, 75 + i * 3], [0, 1], {
      extrapolateRight: "clamp",
    });

  const cylinderAppear = spring({
    frame: local - 60,
    fps,
    config: { damping: 14, stiffness: 110 },
  });

  const arrowToEncoder = interpolate(local, [85, 110], [0, 1], {
    extrapolateRight: "clamp",
  });

  const encoderAppear = spring({
    frame: local - 110,
    fps,
    config: { damping: 14, stiffness: 110 },
  });
  const gatAppear = spring({
    frame: local - 130,
    fps,
    config: { damping: 14, stiffness: 110 },
  });
  const gatPulse = 1 + 0.04 * Math.sin((local - 130) / 6);

  const headsAppear = (i: number) =>
    spring({
      frame: local - 170 - i * 8,
      fps,
      config: { damping: 14, stiffness: 110 },
    });

  const captionOpacity = interpolate(local, [220, 245, 260, 270], [0, 1, 1, 1], {
    extrapolateRight: "clamp",
  });

  const sceneOut = interpolate(local, [255, 270], [1, 0], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        opacity: sceneOut,
      }}
    >
      <SectionHeader text="(C1) Graph World Model" subFrame={local} />

      <svg width="100%" height="100%" viewBox="0 0 1920 1080">
        {/* Edges */}
        {EDGES.map(([a, b], i) => {
          const sa = SERVICES[a];
          const sb = SERVICES[b];
          const pa = nodePos(sa.tier, sa.idx);
          const pb = nodePos(sb.tier, sb.idx);
          const o = edgeAppear(i);
          return (
            <line
              key={i}
              x1={pa.x}
              y1={pa.y + 360}
              x2={pb.x}
              y2={pb.y + 360}
              stroke={palette.borderGray}
              strokeWidth={1.5}
              opacity={o * 0.7}
            />
          );
        })}

        {/* Nodes */}
        {SERVICES.map((s, i) => {
          const p = nodePos(s.tier, s.idx);
          const a = graphAppear(i);
          const r = 22 * a;
          return (
            <g key={i}>
              <circle
                cx={p.x}
                cy={p.y + 360}
                r={r}
                fill={COLORS[i]}
                stroke={palette.borderGray}
                strokeWidth={1}
                opacity={a}
              />
              <text
                x={p.x}
                y={p.y + 360 + 38}
                textAnchor="middle"
                fontFamily="Inter, sans-serif"
                fontSize={13}
                fill={palette.textPrimary}
                opacity={a}
              >
                {s.name}
              </text>
            </g>
          );
        })}

        {/* Logged Transitions cylinder (below graph) */}
        <g
          opacity={cylinderAppear}
          transform={`translate(${380 - 50}, ${810}) scale(${cylinderAppear})`}
        >
          <ellipse cx={50} cy={10} rx={30} ry={8} fill={palette.paleLavender} stroke={palette.borderGray} />
          <rect x={20} y={10} width={60} height={28} fill={palette.paleLavender} stroke={palette.borderGray} />
          <ellipse cx={50} cy={38} rx={30} ry={8} fill={palette.paleLavender} stroke={palette.borderGray} />
          <text
            x={50}
            y={70}
            textAnchor="middle"
            fontFamily="Inter, sans-serif"
            fontSize={16}
            fontWeight={700}
            fill={palette.textPrimary}
          >
            Logged Transitions
          </text>
        </g>

        {/* Arrow: cylinder → Encoder */}
        <line
          x1={460}
          y1={840}
          x2={760 - 10}
          y2={650}
          stroke={palette.borderGray}
          strokeWidth={2}
          opacity={arrowToEncoder}
          markerEnd="url(#arrow)"
        />

        <defs>
          <marker
            id="arrow"
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

        {/* Node Encoder box */}
        <g opacity={encoderAppear} transform={`translate(760, 580) scale(${encoderAppear})`}>
          <rect
            x={0}
            y={0}
            width={220}
            height={120}
            rx={10}
            fill={palette.paleLavender}
            stroke={palette.borderGray}
          />
          <text
            x={110}
            y={50}
            textAnchor="middle"
            fontSize={22}
            fontWeight={700}
            fontFamily="Inter, sans-serif"
            fill={palette.textPrimary}
          >
            Node Encoder
          </text>
          <text
            x={110}
            y={80}
            textAnchor="middle"
            fontSize={14}
            fontStyle="italic"
            fill={palette.textSecondary}
          >
            embeds state + action
          </text>
        </g>

        {/* GAT Layers */}
        <g
          opacity={gatAppear}
          transform={`translate(1050, 580) scale(${gatAppear * gatPulse})`}
        >
          <rect
            x={0}
            y={0}
            width={220}
            height={120}
            rx={10}
            fill={palette.paleLavender}
            stroke={palette.borderGray}
          />
          <text
            x={110}
            y={70}
            textAnchor="middle"
            fontSize={26}
            fontWeight={700}
            fontFamily="Inter, sans-serif"
            fill={palette.textPrimary}
          >
            GAT Layers
          </text>
          {/* circular self-loop arrow */}
          <path
            d="M 200 30 a 18 18 0 1 1 -8 -8"
            fill="none"
            stroke={palette.borderGray}
            strokeWidth={2}
            markerEnd="url(#arrow)"
          />
        </g>

        {/* Three Heads */}
        {[
          { label: "Dynamics Head", out: "→ next state",     fill: palette.paleMint,  y: 460 },
          { label: "Reward Head",   out: "→ reward",         fill: palette.paleYellow, y: 600 },
          { label: "Constraint Head", out: "→ violation risk", fill: palette.paleCoral, y: 740 },
        ].map((h, i) => {
          const o = headsAppear(i);
          return (
            <g key={i} opacity={o} transform={`translate(1380, ${h.y}) scale(${o})`}>
              <rect
                x={0}
                y={0}
                width={260}
                height={100}
                rx={10}
                fill={h.fill}
                stroke={palette.borderGray}
              />
              <text
                x={130}
                y={42}
                textAnchor="middle"
                fontSize={20}
                fontWeight={700}
                fontFamily="Inter, sans-serif"
                fill={palette.textPrimary}
              >
                {h.label}
              </text>
              <text
                x={130}
                y={75}
                textAnchor="middle"
                fontSize={15}
                fontStyle="italic"
                fill={palette.textSecondary}
              >
                {h.out}
              </text>
            </g>
          );
        })}

        {/* GAT → heads arrows */}
        {[510, 650, 790].map((y, i) => (
          <line
            key={i}
            x1={1280}
            y1={640}
            x2={1380 - 8}
            y2={y}
            stroke={palette.borderGray}
            strokeWidth={1.5}
            opacity={headsAppear(i)}
            markerEnd="url(#arrow)"
          />
        ))}
      </svg>

      {/* Caption */}
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
        Learn how scaling actions propagate through the service graph.
      </div>
    </div>
  );
};

const SectionHeader: React.FC<{ text: string; subFrame: number }> = ({
  text,
  subFrame,
}) => {
  const opacity = interpolate(subFrame, [0, 15, 255, 270], [0, 1, 1, 0], {
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
