export const speakPrediction = (text: string) => {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'es-AR';
  console.log('Speaking:', text);
  speechSynthesis.speak(utterance);
};