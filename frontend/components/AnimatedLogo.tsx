'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'

export default function AnimatedLogo() {
  const [hasAnimated, setHasAnimated] = useState(false)

  useEffect(() => {
    // Mark as animated after sequence completes
    const timer = setTimeout(() => setHasAnimated(true), 3500)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="flex items-center justify-center gap-4 md:gap-6">
      {/* Snake slithers in from left with S-curve motion (Chamber of Secrets style) */}
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
          // Add subtle vertical oscillation during horizontal movement (slithering effect)
          animate={!hasAnimated ? {
            y: [0, -8, 0, 8, 0],
          } : {
            y: 0
          }}
          transition={{
            duration: 2,
            times: [0, 0.25, 0.5, 0.75, 1],
            ease: "easeInOut"
          }}
        >
          <Image
            src="/logo-snake.png"
            alt="KARGA Snake"
            width={200}
            height={200}
            className="w-24 h-24 md:w-32 md:h-32 lg:w-40 lg:h-40"
            priority
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
        <Image
          src="/logo-bar.png"
          alt="KARGA Divider"
          width={20}
          height={120}
          className="h-20 w-auto md:h-24 lg:h-28"
          priority
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
        <Image
          src="/logo-name.png"
          alt="KARGA"
          width={400}
          height={120}
          className="w-48 h-auto md:w-64 lg:w-80"
          priority
        />
      </motion.div>
    </div>
  )
}
