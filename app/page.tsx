"use client";

import React, { useEffect, useState } from "react";

const images = [
  "/aristotle.jpg", // Replace these with actual paths to your images
  "/hume.jpg",
  "/kant.jpg",
  "/mill.jpg",
  "/schopenhauer.jpg",
];

export default function Home() {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [fade, setFade] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setFade(false);
      setTimeout(() => {
        setCurrentImageIndex((prevIndex) => (prevIndex + 1) % images.length);
        setFade(true);
      }, 500); // Half the interval to allow for fade-out and then fade-in
    }, 3000); // Change image every 3000 milliseconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative min-h-screen bg-black">
      <style jsx>{`
        .image-slash {
          clip-path: polygon(0% 8%, 100% 8%, 100% 100%, 0% 100%);
        }
      `}</style>
      <div
        className={`absolute inset-0 transition-opacity duration-1000 ease-in-out image-slash`}
        style={{
          backgroundImage: `url(${images[currentImageIndex]})`,
          backgroundSize: "cover",
          opacity: fade ? 1 : 0,
        }}
      />
      <h1 className="z-10 relative text-white text-center mt-4 text-4xl">
        Welcome to Philosophy
      </h1>
    </div>
  );
}
