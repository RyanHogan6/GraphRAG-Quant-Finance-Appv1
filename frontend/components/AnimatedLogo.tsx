'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'

export default function AnimatedLogo() {
  const [hasAnimated, setHasAnimated] = useState(false)

  useEffect(() => {
    // Mark as animated after first render
    const timer = setTimeout(() => setHasAnimated(true), 2000)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="flex items-center justify-center">
      {/* Snake slithers in from left */}
      <motion.div
        initial={{ x: -300, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{
          type: "spring",
          damping: 15,
          stiffness: 50,
          duration: 1.5,
          delay: 0.3
        }}
        className="relative"
      >
        <Image
          src="/thumbnail3.jpg"
          alt="KARGA Logo"
          width={800}
          height={200}
          className="w-full max-w-[600px] md:max-w-[800px] h-auto"
          priority
        />
      </motion.div>
    </div>
  )
}
