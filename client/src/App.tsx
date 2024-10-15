import { useRef, useEffect } from 'react';
import './App.css';
import { Holistic } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: true,
      smoothSegmentation: true,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults(onResults);

    const camera = new Camera(videoRef.current!, {
      onFrame: async () => {
        await holistic.send({ image: videoRef.current! });
      },
      width: 640,
      height: 480,
    });
    camera.start();

    function onResults(results: any) {
      if (!canvasRef.current || !videoRef.current) return;
      const canvasCtx = canvasRef.current.getContext('2d');
      canvasCtx?.save();
      canvasCtx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      // Draw the video
      canvasCtx?.drawImage(
        results.image,
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );

      // Draw face landmarks
      if (results.faceLandmarks) {
        drawLandmarks(canvasCtx!, results.faceLandmarks, { color: 'red', lineWidth: 1 });
      }

      // Draw hand landmarks
      if (results.rightHandLandmarks) {
        drawLandmarks(canvasCtx!, results.rightHandLandmarks, { color: 'blue', lineWidth: 2 });
      }
      if (results.leftHandLandmarks) {
        drawLandmarks(canvasCtx!, results.leftHandLandmarks, { color: 'blue', lineWidth: 2 });
      }

      canvasCtx?.restore();
    }

    function drawLandmarks(ctx: CanvasRenderingContext2D, landmarks: any, style: any) {
      for (let i = 0; i < landmarks.length; i++) {
        const x = landmarks[i].x * canvasRef.current!.width;
        const y = landmarks[i].y * canvasRef.current!.height;
        ctx.beginPath();
        ctx.arc(x, y, style.lineWidth, 0, 2 * Math.PI);
        ctx.fillStyle = style.color;
        ctx.fill();
      }
    }
  }, []);

  return (
    <>
      <video ref={videoRef} />
      <canvas ref={canvasRef} width="640" height="480" className="output_canvas"></canvas>
    </>
  );
}

export default App;
