import { Results, NormalizedLandmarkList } from '@mediapipe/holistic';
import { drawLandmarks } from './drawUtils'

export const onResults = (results: Results, setKeypointsSequence: (seq: any) => void, canvasRef: React.RefObject<HTMLCanvasElement>) => {
  const keypoints = extractKeypoints(results);
  const flattenedKeypoints = keypoints.flat();

  if (flattenedKeypoints.length !== 96) {
    console.error('Error: The frame does not contain 96 keypoints.');
    return;
  }

  setKeypointsSequence((prev: number[][]) => [...prev, flattenedKeypoints]);
  drawLandmarks(results, canvasRef);
};

const extractKeypoints = (results: Results): number[][] => {
  const keypoints: number[][] = [];

  keypoints.push(results.leftHandLandmarks ? flattenLandmarks(results.leftHandLandmarks) : Array(42).fill(0));
  keypoints.push(results.rightHandLandmarks ? flattenLandmarks(results.rightHandLandmarks) : Array(42).fill(0));

  if (results.faceLandmarks) {
    const faceKeypoints = filterFaceLandmarks(results.faceLandmarks);
    keypoints.push(flattenLandmarks(faceKeypoints));
  } else {
    keypoints.push(Array(12).fill(0));
  }

  return keypoints;
};

const flattenLandmarks = (landmarks: NormalizedLandmarkList): number[] => {
  return landmarks.map(landmark => [landmark.x, landmark.y]).flat();
};

const filterFaceLandmarks = (landmarks: NormalizedLandmarkList): NormalizedLandmarkList => {
  const requiredLandmarks = [1, 33, 152, 234, 263, 454];
  return requiredLandmarks.map(index => landmarks[index]);
};
