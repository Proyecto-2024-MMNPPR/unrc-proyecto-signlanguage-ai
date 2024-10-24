import './App.css';
import { useState, useRef } from 'react';
import background from './assets/background.png';

import { useHolistic } from './hooks/useHolistic';
import { speakPrediction } from './utils/speakPrediction';
import { fetchBackendPrediction, fetchOpenAiResponse } from './utils/fetchBackend';

import { Header } from './components/Header';
import { SendButton } from './components/SendButton';
import { VideoAndCanvas } from './components/VideoAndCanvas';
import { PredictionResult } from './components/PredictionResult';
import { linspace } from './utils/linspace';

function App() {
  const [keypointsSequence, setKeypointsSequence] = useState<number[][]>([]);
  const [prediction, setPrediction] = useState<string>('');
  const [openAiResponse, setOpenAiResponse] = useState<string>('');
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const numRepresentativeFrames = 5;

  // Initialize Holistic and Camera
  useHolistic(videoRef, canvasRef, setKeypointsSequence);

  // Send sequence of keypoints to the backend
  const sendSequenceToBackend = async () => {
    if (keypointsSequence.length < numRepresentativeFrames) {
      console.log("Sequence is too short. Accumulated frames:", keypointsSequence.length);
      return;
    }

    const selectedFrames = linspace(0, keypointsSequence.length - 1, numRepresentativeFrames);
    const finalSamples = selectedFrames.map(index => keypointsSequence[index]);

    try {
      const backendResponse = await fetchBackendPrediction(finalSamples);
      const openAiData = await fetchOpenAiResponse(backendResponse.prediction);

      // Update states and handle TTS
      setPrediction(backendResponse.prediction);
      setOpenAiResponse(openAiData.ai_response);
      setKeypointsSequence([]);

      speakPrediction(openAiData.ai_response);
    } catch (error) {
      console.error('Error sending sequence:', error);
    }
  };

  return (
    <div className='h-screen w-screen flex flex-col justify-center items-center gap-4 px-8 bg-cover bg-center' style={{ backgroundImage: `url("${background}")` }}>
      <div className='w-full p-4 bg-gray-700/30 border border-gray-500 backdrop-blur-md rounded-lg'>
        <Header />
        <VideoAndCanvas videoRef={videoRef} canvasRef={canvasRef} />
        <SendButton onClick={sendSequenceToBackend} />
        {prediction && <PredictionResult prediction={openAiResponse} />}
      </div>
    </div>
  );
}

export default App;
