import { Results, NormalizedLandmarkList } from '@mediapipe/holistic';

export const drawLandmarks = (results: Results, canvasRef: React.RefObject<HTMLCanvasElement>) => {
  if (!canvasRef.current) return;
  const canvasCtx = canvasRef.current.getContext('2d');
  if (!canvasCtx) return;

  // Clear the canvas
  canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  canvasCtx.drawImage(results.image as CanvasImageSource, 0, 0, canvasRef.current.width, canvasRef.current.height);

  // Draw only the 5 most representative face landmarks
  if (results.faceLandmarks) {
    const faceLandmarks = filterFaceLandmarks(results.faceLandmarks);
    drawLandmarkSet(canvasCtx, faceLandmarks, 'red');
  }

  // Draw hand landmarks
  if (results.leftHandLandmarks) drawLandmarkSet(canvasCtx, results.leftHandLandmarks, 'blue');
  if (results.rightHandLandmarks) drawLandmarkSet(canvasCtx, results.rightHandLandmarks, 'blue');
};

// Filter to only take the 5 most representative face landmarks
const filterFaceLandmarks = (landmarks: NormalizedLandmarkList) => {
  const requiredLandmarksIndices = [1, 33, 152, 234, 263, 454];
  return requiredLandmarksIndices.map(index => landmarks[index]);
};

// Draw a set of landmarks on the canvas
const drawLandmarkSet = (ctx: CanvasRenderingContext2D, landmarks: any, color: string) => {
  landmarks.forEach((landmark: any) => drawPoint(ctx, landmark, color, 2));
};

// Draw individual point on the canvas
const drawPoint = (ctx: CanvasRenderingContext2D, landmark: { x: number, y: number }, color: string, size: number) => {
  const x = landmark.x * ctx.canvas.width;
  const y = landmark.y * ctx.canvas.height;
  ctx.beginPath();
  ctx.arc(x, y, size, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
};
