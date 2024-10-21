import { useRef, useEffect, useState } from 'react';
import { Holistic } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';
import background from './assets/background.png';
import unrcLogotype from './assets/unrc-logotype.png';
import './App.css';

function App() {
  // States
  const [currentTime, setCurrentTime] = useState('');

  // References
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Secondary effects
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

  useEffect(() => {
    const updateCurrentTime = () => {
      const now = new Date();
      const hours = now.getHours() % 12 || 12;
      const minutes = now.getMinutes().toString().padStart(2, '0');
      const ampm = now.getHours() >= 12 ? 'PM' : 'AM';
      setCurrentTime(`${hours}:${minutes} ${ampm}`);
    };

    updateCurrentTime();
    const intervalId = setInterval(updateCurrentTime, 60000);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className='h-screen w-screen flex flex-col justify-center items-center gap-4 px-8 bg-cover bg-center' style={{ backgroundImage: `url("${background}")` }}>
      <div className='w-full p-4 bg-gray-700/30 border border-gray-500 backdrop-blur-md rounded-lg'>
        <div className="flex items-center justify-between">
          <div>
            <span className="text-lg font-light text-gray-400">{currentTime}</span>
            <span className="mx-2 text-lg text-gray-400">|</span>
            <span className="text-lg font-medium text-white">✋ Digalo - Aplicación de Lenguaje de Señas Argentina</span>
          </div>
          <img src={unrcLogotype} alt="UNRC Logotype" className="h-14" />
        </div>

        <div className='flex justify-center items-center gap-4 mt-2'>
          <div className="w-1/2 h-auto rounded-lg">
            <video
              ref={videoRef}
              className='w-full h-auto rounded-lg'
              style={{ objectFit: 'cover' }}
            />
          </div>
          <div className="w-1/2 h-auto rounded-lg">
            <canvas
              ref={canvasRef}
              width="640"
              height="480"
              className="w-full h-auto rounded-lg"
              style={{ objectFit: "cover" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
