import { Composition } from "remotion";
import { NetDream } from "./NetDream";

const FPS = 30;
const TOTAL_FRAMES = 30 * FPS; // 30 seconds at 30 fps

export const Root: React.FC = () => {
  return (
    <Composition
      id="NetDream"
      component={NetDream}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={1920}
      height={1080}
    />
  );
};
