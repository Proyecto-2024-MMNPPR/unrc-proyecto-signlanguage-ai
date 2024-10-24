import { useEffect, useState } from "react";
import unrcLogotype from "../assets/unrc-logotype.png"
import { getCurrentTime } from "../utils/getCurrentTime";

export const Header = () => {
  const [currentTime, setCurrentTime] = useState<string>('');

  // Update current time every minute
  useEffect(() => {
    const intervalId = setInterval(() => setCurrentTime(getCurrentTime()), 60000);
    setCurrentTime(getCurrentTime());

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="flex items-center justify-between">
      <div>
        <span className="text-lg font-light text-gray-400">{currentTime}</span>
        <span className="mx-2 text-lg text-gray-400">|</span>
        <span className="text-lg font-medium text-white">✋ Digalo - Aplicación de Lenguaje de Señas Argentina</span>
      </div>
      <img src={unrcLogotype} alt="UNRC Logotype" className="h-14" />
    </div>
  )
};
