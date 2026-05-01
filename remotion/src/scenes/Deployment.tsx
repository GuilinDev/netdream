import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { palette } from "../lib/palette";
import { timing, sceneFrame } from "../lib/timing";
import { PanelFrame } from "../components/PanelFrame";

// 22–27 s (frames 660-810) — Module C3: Online Deployment
//
// Beats:
//   0-30  Cluster racks appear; pods start scaling up/down
//   30-90 Selected action arrow flows in; pods react
//   90-150 Loop arrow: Metrics → Controller (the closed loop)

export const Deployment: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = sceneFrame(frame, timing.c3Start);

  const controllerAppear = spring({
    frame: local - 5,
    fps,
    config: { damping: 13, stiffness: 110 },
  });
  const clusterAppear = spring({
    frame: local - 20,
    fps,
    config: { damping: 13, stiffness: 110 },
  });
  const metricsAppear = spring({
    frame: local - 35,
    fps,
    config: { damping: 13, stiffness: 110 },
  });

  const arrowProgress = (start: number, end: number) =>
    interpolate(local, [start, end], [0, 1], { extrapolateRight: "clamp" });

  // Pods inside the cluster: animate scale-up
  const podGrowth = (i: number) => {
    const startFrame = 50 + i * 6;
    return spring({
      frame: local - startFrame,
      fps,
      config: { damping: 15, stiffness: 90 },
    });
  };

  const captionOpacity = interpolate(local, [110, 130, 140, 150], [0, 1, 1, 0], {
    extrapolateRight: "clamp",
  });

  const sceneOut = interpolate(local, [135, 150], [1, 0], {
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ position: "absolute", inset: 0, opacity: sceneOut }}>
      <PanelFrame
        variant="mint"
        localFrame={local}
        fadeStart={0}
        fadeEnd={15}
        fadeOutStart={135}
        fadeOutEnd={150}
      />
      <SectionHeader text="(C3) Online Deployment on Cluster" subFrame={local} />

      <svg width="1920" height="1080" viewBox="0 0 1920 1080" style={{ position: "absolute", top: 0, left: 0 }}>
        <defs>
          <marker
            id="arrow3"
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

        {/* === Kubernetes Controller (left) === */}
        <g
          opacity={controllerAppear}
          transform={`translate(280, 400) scale(${controllerAppear})`}
        >
          <rect
            x={0}
            y={0}
            width={320}
            height={130}
            rx={10}
            fill={palette.paleBlue}
            stroke={palette.borderGray}
          />
          <text
            x={160}
            y={55}
            textAnchor="middle"
            fontSize={26}
            fontWeight={700}
            fontFamily="Inter, sans-serif"
            fill={palette.textPrimary}
          >
            Kubernetes Controller
          </text>
          <text
            x={160}
            y={90}
            textAnchor="middle"
            fontSize={16}
            fontStyle="italic"
            fill={palette.textSecondary}
          >
            applies scaling actions
          </text>
        </g>

        {/* === Cluster (middle) === */}
        <g
          opacity={clusterAppear}
          transform={`translate(800, 350) scale(${clusterAppear})`}
        >
          <rect
            x={0}
            y={0}
            width={420}
            height={240}
            rx={10}
            fill="#FFFFFF"
            stroke={palette.borderGray}
          />
          <text
            x={210}
            y={42}
            textAnchor="middle"
            fontSize={26}
            fontWeight={700}
            fontFamily="Inter, sans-serif"
            fill={palette.textPrimary}
          >
            Cluster
          </text>
          {/* Server racks (3) with pod squares inside that grow */}
          {[0, 1, 2].map((rackIdx) => {
            const rackX = 50 + rackIdx * 120;
            return (
              <g key={rackIdx}>
                {/* Rack body */}
                <rect
                  x={rackX}
                  y={70}
                  width={90}
                  height={150}
                  rx={6}
                  fill={palette.paleLavender}
                  stroke={palette.borderGray}
                  strokeWidth={1.5}
                />
                {/* Pods (rows) — scale up over time */}
                {[0, 1, 2, 3, 4].map((podIdx) => {
                  const a = podGrowth(rackIdx * 5 + podIdx);
                  return (
                    <rect
                      key={podIdx}
                      x={rackX + 8}
                      y={80 + podIdx * 26}
                      width={74 * a}
                      height={20}
                      rx={3}
                      fill={palette.accentBlue}
                      opacity={a * 0.55}
                    />
                  );
                })}
              </g>
            );
          })}
        </g>

        {/* === Metrics (right) === */}
        <g
          opacity={metricsAppear}
          transform={`translate(1480, 400) scale(${metricsAppear})`}
        >
          <rect
            x={0}
            y={0}
            width={260}
            height={130}
            rx={10}
            fill={palette.palePeach}
            stroke={palette.borderGray}
          />
          <text
            x={130}
            y={75}
            textAnchor="middle"
            fontSize={28}
            fontWeight={700}
            fontFamily="Inter, sans-serif"
            fill={palette.textPrimary}
          >
            Metrics
          </text>
        </g>

        {/* Arrows: Controller → Cluster → Metrics */}
        <line
          x1={600}
          y1={465}
          x2={800 - 8}
          y2={465}
          stroke={palette.borderGray}
          strokeWidth={2.5}
          markerEnd="url(#arrow3)"
          opacity={arrowProgress(40, 60)}
        />
        <line
          x1={1220}
          y1={465}
          x2={1480 - 8}
          y2={465}
          stroke={palette.borderGray}
          strokeWidth={2.5}
          markerEnd="url(#arrow3)"
          opacity={arrowProgress(55, 75)}
        />

        {/* Curved feedback arrow: Metrics → Controller */}
        <path
          d={`M 1610 530 Q 1610 700, 1300 700 T 440 530`}
          fill="none"
          stroke={palette.borderGray}
          strokeWidth={2}
          markerEnd="url(#arrow3)"
          opacity={arrowProgress(80, 110)}
          strokeDasharray="8 4"
        />
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
        Closed-loop deployment on AWS EKS — every 5 seconds.
      </div>
    </div>
  );
};

const SectionHeader: React.FC<{ text: string; subFrame: number }> = ({
  text,
  subFrame,
}) => {
  const opacity = interpolate(subFrame, [0, 15, 135, 150], [0, 1, 1, 0], {
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        position: "absolute",
        top: 60, zIndex: 10,
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
