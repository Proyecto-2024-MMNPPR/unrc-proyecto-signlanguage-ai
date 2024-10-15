import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setIsCameraOn(true);
      })
      .catch((err) => {
        console.error('Error accessing the camera:', err);
      });

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <>
      <video ref={videoRef} autoPlay className="video-feed" />
      {!isCameraOn && <p>Waiting for camera access...</p>}
    </>
  );
}

export default App;
