import { useEffect, useRef, useState } from 'react';
import axios from 'axios';

function App() {
  const videoRef = useRef<any>(null);
  const [detectedMessage, setDetectedMessage] = useState('');
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;

          const interval = setInterval(() => {
            captureFrame();
          }, 1000);

          return () => clearInterval(interval);
        })
        .catch(err => {
          console.error("Error accessing the camera: ", err);
        });
    }
  }, []);

  const captureFrame = async () => {
    const video = videoRef.current;
    const canvas: any = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    const imgData = canvas.toDataURL('image/jpeg');
    const imgBase64 = imgData.split(',')[1];

    try {
      console.log("Sending frame to backend...");
      const response = await axios.post('http://localhost:5001/api/process-frame', { image: imgBase64 });

      console.log(response.data);

      if (response.data.message) {
        setDetectedMessage(response.data.message);
        setConfidence(response.data.confidence);
      }
    } catch (error) {
      console.error("Error processing the frame: ", error);
    }
  };

  return (
    <div className="App">
      <h1>Sign Language Detector</h1>
      <video ref={videoRef} autoPlay style={{ width: '100%', height: 'auto' }}></video>
      {detectedMessage && (
        <div>
          <h2>Detected Letter: {detectedMessage}</h2>
          <p>Confidence: {confidence.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
