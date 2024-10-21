import { useRef, useEffect, useState } from 'react';
import { Holistic } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';
import background from './assets/background.png';
import unrcLogotype from './assets/unrc-logotype.png';
import axios from 'axios';
import './App.css';

function App() {
  // States
  const [currentTime, setCurrentTime] = useState('');
  const [keypointsSequence, setKeypointsSequence] = useState([]); // Estado para almacenar la secuencia de keypoints
  const [prediction, setPrediction] = useState(''); // Estado para la predicción del backend

  // References
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const sendSequenceToBackend = async (sequence) => {
    console.log("Sending sequence to backend...", sequence);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sequence: sequence,
        }),
      })
        .then(response => response.json())
        .then(data => console.log('Prediction:', data))
        .catch(error => console.error('Error:', error));
    } catch (error) {
      console.error('Error sending sequence:', error);
    }
  };


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

      // Draw hand landmarks and collect keypoints
      let keypoints = [];
      let rightHandKeypoints = [];
      let leftHandKeypoints = [];

      if (results.rightHandLandmarks) {
        // Flatten right hand landmarks (x, y coordinates)
        rightHandKeypoints = results.rightHandLandmarks.map((point: any) => [point.x, point.y]).flat();
      }

      if (results.leftHandLandmarks) {
        // Flatten left hand landmarks (x, y coordinates)
        leftHandKeypoints = results.leftHandLandmarks.map((point: any) => [point.x, point.y]).flat();
      }

      // Rellenar con ceros si falta una mano
      if (rightHandKeypoints.length === 0) {
        rightHandKeypoints = Array(42).fill(0); // Rellenar con 42 ceros
      }
      if (leftHandKeypoints.length === 0) {
        leftHandKeypoints = Array(42).fill(0); // Rellenar con 42 ceros
      }

      keypoints = rightHandKeypoints.concat(leftHandKeypoints);

      // Agregar keypoints a la secuencia si hay datos (84 features per frame)
      if (keypoints.length === 84) {
        setKeypointsSequence((prevSequence: any) => {
          const newSequence = [...prevSequence, keypoints].slice(-30); // Mantener longitud máxima de 30 frames
          if (newSequence.length === 30) {
            sendSequenceToBackend(newSequence.flat()); // Enviar la secuencia cuando esté completa
          }
          return newSequence;
        });
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
        {prediction && (
          <div className="mt-4 text-white text-lg">
            Prediction: {prediction}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
