import React, { useState, useEffect } from 'react';

interface TypewriterProps {
  text: string;
  speed?: number; // Speed in milliseconds per character
  className?: string;
  onDone?: () => void; // Callback when typing is done
}

const Typewriter: React.FC<TypewriterProps> = ({ text, speed = 50, className = '', onDone }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, speed);
      return () => clearTimeout(timeout);
    } else {
      if (onDone) {
        onDone();
      }
    }
  }, [currentIndex, text, speed, onDone]);

  // Reset when text changes
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
  }, [text]);

  return <span className={className}>{displayedText}</span>;
};

export default Typewriter;
