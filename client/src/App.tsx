import { useRef, useEffect, useState } from 'react';
import { Holistic, Results } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';
import background from './assets/background.png';
import unrcLogotype from './assets/unrc-logotype.png';
import './App.css';

function App() {
  // States
  const [currentTime, setCurrentTime] = useState<string>('');
  const [keypointsSequence, setKeypointsSequence] = useState<number[][]>([]);
  const [prediction, setPrediction] = useState<string>('');

  // References
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Function to play TTS based on the prediction
  const speakPrediction = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'es-AR'; // Lenguaje en español argentino
    speechSynthesis.speak(utterance);
  };

  const sendSequenceToBackend = async () => {
    if (keypointsSequence.length < 30) {
      console.log("Sequence is too short. Accumulated frames:", keypointsSequence.length);
      return;
    }

    console.log("Sending sequence to backend...", keypointsSequence);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sequence: keypointsSequence.slice(0, 30),
        }),
      });

      const data = await response.json();
      console.log('Prediction:', data);
      setPrediction(data.prediction);
    } catch (error) {
      console.error('Error sending sequence:', error);
    }
  };

  const normalizeKeypoints = (landmarks: { x: number, y: number }[]): number[] => {
    const xCoords = landmarks.map(landmark => landmark.x);
    const yCoords = landmarks.map(landmark => landmark.y);

    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);

    const normalizedLandmarks = landmarks.map(landmark => {
      const normalizedX = (landmark.x - minX) / (maxX - minX || 1);
      const normalizedY = (landmark.y - minY) / (maxY - minY || 1);
      return [normalizedX, normalizedY];
    }).flat();

    return normalizedLandmarks;
  };

  const onResults = (results: Results) => {
    const keypoints: number[][] = [];

    if (results.faceLandmarks) {
      const normalizedFaceKeypoints = normalizeKeypoints(results.faceLandmarks.slice(0, 21));
      keypoints.push(normalizedFaceKeypoints);
    }

    if (results.rightHandLandmarks) {
      const normalizedRightHandKeypoints = normalizeKeypoints(results.rightHandLandmarks);
      keypoints.push(normalizedRightHandKeypoints);
    }

    if (results.leftHandLandmarks) {
      const normalizedLeftHandKeypoints = normalizeKeypoints(results.leftHandLandmarks);
      keypoints.push(normalizedLeftHandKeypoints);
    }

    if (results.poseLandmarks) {
      const normalizedPoseKeypoints = normalizeKeypoints(results.poseLandmarks);
      keypoints.push(normalizedPoseKeypoints);
    }

    let flattenedKeypoints = keypoints.flat();

    if (flattenedKeypoints.length < 84) {
      const missingPoints = 84 - flattenedKeypoints.length;
      flattenedKeypoints = [...flattenedKeypoints, ...Array(missingPoints).fill(0)];
    }

    if (flattenedKeypoints.length > 84) {
      flattenedKeypoints = flattenedKeypoints.slice(0, 84);
    }

    if (flattenedKeypoints.length !== 84) {
      console.error('Error: El frame no tiene 84 características.');
      return;
    }

    setKeypointsSequence((prevSequence) => [...prevSequence, flattenedKeypoints]);

    drawLandmarks(results);
  };

  const drawLandmarks = (results: Results) => {
    if (!canvasRef.current || !videoRef.current) return;
    const canvasCtx = canvasRef.current.getContext('2d');
    canvasCtx?.save();
    canvasCtx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Draw the video
    canvasCtx?.drawImage(
      results.image as CanvasImageSource,
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    // Draw face landmarks
    if (results.faceLandmarks) {
      results.faceLandmarks.forEach((landmark) => {
        drawPoint(canvasCtx!, landmark, 'red', 1);
      });
    }

    // Draw hand landmarks
    if (results.rightHandLandmarks) {
      results.rightHandLandmarks.forEach((landmark) => {
        drawPoint(canvasCtx!, landmark, 'blue', 2);
      });
    }
    if (results.leftHandLandmarks) {
      results.leftHandLandmarks.forEach((landmark) => {
        drawPoint(canvasCtx!, landmark, 'blue', 2);
      });
    }

    canvasCtx?.restore();
  };

  const drawPoint = (ctx: CanvasRenderingContext2D, landmark: { x: number, y: number }, color: string, size: number) => {
    const x = landmark.x * canvasRef.current!.width;
    const y = landmark.y * canvasRef.current!.height;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
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

  // Effect to trigger TTS when a new prediction is made
  useEffect(() => {
    if (prediction) {
      speakPrediction(prediction);
    }
  }, [prediction]);

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

        <button
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          onClick={sendSequenceToBackend}
        >
          Enviar Secuencia
        </button>

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
