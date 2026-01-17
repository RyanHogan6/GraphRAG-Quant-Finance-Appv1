'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

export default function AnimatedLogo() {
  const [hasAnimated, setHasAnimated] = useState(false)

  useEffect(() => {
    // Mark as animated after sequence completes
    const timer = setTimeout(() => setHasAnimated(true), 3500)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="flex items-center justify-center" style={{ gap: '16px' }}>
      {/* Raven swoops in from left with fluid motion */}
      <motion.div
        initial={{ x: -400, opacity: 0, rotate: -10 }}
        animate={{
          x: 0,
          opacity: 1,
          rotate: 0,
          // Create slithering S-curve motion
          transition: {
            x: {
              type: "spring",
              damping: 20,
              stiffness: 40,
              duration: 2,
            },
            opacity: {
              duration: 0.5,
              delay: 0.2
            },
            rotate: {
              duration: 2,
              ease: [0.43, 0.13, 0.23, 0.96] // Custom ease for smooth rotation
            }
          }
        }}
        className="relative"
      >
        <motion.div
          // Add serpentine undulation - snake waves side to side as it moves
          animate={!hasAnimated ? {
            y: [0, -8, 0, 8, 0],
            rotateZ: [0, -3, 0, 3, 0, -2, 0],
            skewY: [0, 2, 0, -2, 0, 1, 0],
            scaleX: [1, 1.02, 1, 0.98, 1, 1.01, 1],
          } : {
            y: 0,
            rotateZ: 0,
            skewY: 0,
            scaleX: 1
          }}
          transition={{
            duration: 2,
            times: [0, 0.15, 0.3, 0.5, 0.65, 0.85, 1],
            ease: "easeInOut",
            rotateZ: {
              duration: 2,
              ease: [0.45, 0.05, 0.55, 0.95] // Smooth sine wave
            },
            skewY: {
              duration: 2,
              ease: "easeInOut"
            }
          }}
        >
          <img
            src="/logo-raven.png"
            alt="KARGA Raven"
            className="object-contain"
            style={{ width: '320px', height: '320px' }}
          />
        </motion.div>
      </motion.div>

      {/* Vertical bar drops in after snake arrives */}
      <motion.div
        initial={{ y: -100, opacity: 0, scaleY: 0 }}
        animate={{
          y: 0,
          opacity: 1,
          scaleY: 1,
          transition: {
            delay: 1.8, // Wait for snake to mostly arrive
            duration: 0.6,
            type: "spring",
            stiffness: 200,
            damping: 15
          }
        }}
        className="relative"
        style={{ transformOrigin: 'top' }}
      >
        <img
          src="/logo-bar.png"
          alt="KARGA Divider"
          className="w-auto object-contain"
          style={{ height: '320px', width: 'auto' }}
        />
      </motion.div>

      {/* KARGA text fades and scales in last */}
      <motion.div
        initial={{ opacity: 0, scale: 0.8, x: 20 }}
        animate={{
          opacity: 1,
          scale: 1,
          x: 0,
          transition: {
            delay: 2.2, // Wait for bar to drop
            duration: 0.8,
            ease: [0.25, 0.46, 0.45, 0.94] // Smooth ease out
          }
        }}
        className="relative"
      >
        <img
          src="/logo-name.png"
          alt="KARGA"
          className="h-auto object-contain"
          style={{ width: '640px', height: 'auto' }}
        />
      </motion.div>
    </div>
  )
}
