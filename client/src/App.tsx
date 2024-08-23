import { useEffect, useRef, useState } from 'react';
import axios from 'axios';

function App() {
  const videoRef = useRef<any>(null);
  const [lastMessage, setLastMessage] = useState('');
  const [lastMessageTime, setLastMessageTime] = useState(0);
  const [detectedMessage, setDetectedMessage] = useState('');
  const [processedImage, setProcessedImage] = useState<any>(null);

  useEffect(() => {
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
        })
        .catch(err => {
          console.error("Error accessing the camera: ", err);
        });
    }

    const interval = setInterval(captureFrame, 1000);
    return () => clearInterval(interval);
  }, []);

  const vocalize = (text: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'es-ES';
      window.speechSynthesis.speak(utterance);
    } else {
      console.error("Speech Synthesis not supported in this browser.");
    }
  };

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
      const response = await axios.post('/api/process-frame', { image: imgBase64 });

      console.log(response.data);

      setProcessedImage(`data:image/jpeg;base64,${response.data.processed_image}`);

      const currentTime = new Date().getTime();
      const timeDifference = currentTime - lastMessageTime;

      if (response.data.message && (response.data.message !== lastMessage || timeDifference > 5000)) {
        setDetectedMessage(response.data.message);
        setLastMessage(response.data.message);
        setLastMessageTime(currentTime);
        vocalize(response.data.message);
      }
    } catch (error) {
      console.error("Error processing the frame: ", error);
    }
  };

  return (
    <div className="App">
      <h1>Sign Language Detector</h1>
      <video ref={videoRef} autoPlay></video>
      {processedImage && (
        <div>
          <h2>Processed Frame:</h2>
          <img src={processedImage} alt="Processed" />
        </div>
      )}
      {detectedMessage && (
        <div>
          <h2>Detected Message:</h2>
          <p>{detectedMessage}</p>
        </div>
      )}
    </div>
  );
}

export default App;
