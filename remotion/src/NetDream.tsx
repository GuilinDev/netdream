import { Sequence, useCurrentFrame } from "remotion";
import { Background } from "./components/Background";
import { Title } from "./scenes/Title";
import { GraphWorldModel } from "./scenes/GraphWorldModel";
import { ImaginationPlanner } from "./scenes/ImaginationPlanner";
import { Deployment } from "./scenes/Deployment";
import { Outro } from "./scenes/Outro";
import { timing } from "./lib/timing";

// Top-level composition — orchestrates all five scenes onto a 30-second
// timeline. Each scene receives the global frame and computes its own
// local frame relative to its start. This keeps timing logic local to
// each component.

export const NetDream: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <>
      {/* Persistent background through all scenes */}
      <Background />

      {/* Scene 1: Title */}
      <Sequence from={timing.titleStart} durationInFrames={timing.titleEnd - timing.titleStart}>
        <Title />
      </Sequence>

      {/* Scene 2: Graph World Model (C1) */}
      <Sequence from={timing.c1Start} durationInFrames={timing.c1End - timing.c1Start}>
        <GraphWorldModel />
      </Sequence>

      {/* Scene 3: Imagination-Based Planner (C2) */}
      <Sequence from={timing.c2Start} durationInFrames={timing.c2End - timing.c2Start}>
        <ImaginationPlanner />
      </Sequence>

      {/* Scene 4: Online Deployment (C3) */}
      <Sequence from={timing.c3Start} durationInFrames={timing.c3End - timing.c3Start}>
        <Deployment />
      </Sequence>

      {/* Scene 5: Outro / Pareto */}
      <Sequence from={timing.outroStart} durationInFrames={timing.outroEnd - timing.outroStart}>
        <Outro />
      </Sequence>
    </>
  );
};
