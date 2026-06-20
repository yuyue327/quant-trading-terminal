import { useEffect, useState } from 'react';

export const useMouseGlow = () => {
  const [position, setPosition] = useState({ x: -1000, y: -1000 });

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMove);
    return () => window.removeEventListener('mousemove', handleMove);
  }, []);

  return position;
};