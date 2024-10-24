export const fetchBackendPrediction = async (finalSamples: number[][]) => {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ sequence: finalSamples }),
  });
  return response.json();
};

export const fetchOpenAiResponse = async (prediction: string) => {
  const response = await fetch('/api/openai', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ prediction }),
  });
  return response.json();
};
